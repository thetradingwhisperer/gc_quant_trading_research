import toml
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
import glob
from tqdm import tqdm
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the configuration
config = toml.load("ml_experiment_config.toml")

# Set up the model
MODEL = config['model']['name']
EMBEDDING_SIZE = config['data']['embedding_size']

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", MODEL)

device = torch.device(config['model']['device'] if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

# Set up image transformation
transform_image = T.Compose([
    T.ToTensor(),
    T.Resize(tuple(config['data']['image_size'])),
    T.Normalize(config['preprocessing']['normalization_mean'], 
                config['preprocessing']['normalization_std'])
])

def load_image(img: str) -> torch.Tensor:
    """Load an image and return a tensor that can be used as an input to DINOv2."""
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img

def compute_embeddings(files: list) -> dict:
    """Create an index that contains all of the images in the specified list of files."""
    all_embeddings = {}
    with torch.no_grad():
        for file in tqdm(files, desc="Computing embeddings"):
            embeddings = dinov2_vits14(load_image(file).to(device))
            all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
    
    with open(config['embeddings']['output_file'], "w") as f:
        json.dump(all_embeddings, f)
    
    return all_embeddings

# Load and prepare data
ROOT_DIR = config['paths']['root_dir']
labels = {}

for folder in os.listdir(ROOT_DIR):
    if folder == config['paths']['test_folder']:
        continue
    try:
        for file in os.listdir(os.path.join(ROOT_DIR, folder)):
            if file.endswith(".png"):
                full_name = os.path.join(ROOT_DIR, folder, file)
                labels[full_name] = folder
    except:
        pass

files = list(labels.keys())

# Compute embeddings
embeddings = compute_embeddings(files)

# Prepare data for training
y = [labels[file] for file in files]
embedding_list = list(embeddings.values())
embedding_arr = np.array(embedding_list).reshape(-1, EMBEDDING_SIZE)

# Define classifier fitting functions
def create_pipeline(classifier, use_pca=True):
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=config['pca']['n_components'], random_state=config['pca']['random_state'])))
    steps.append(('classifier', classifier))
    return Pipeline(steps)

def fit_svm(embedding_list, y, use_pca=True):
    clf = svm.SVC(gamma=config['svm']['gamma'])
    pipeline = create_pipeline(clf, use_pca)
    pipeline.fit(np.array(embedding_list).reshape(-1, EMBEDDING_SIZE), y)
    return pipeline

def fit_rf(embedding_list, y, use_pca=True):
    rf = RandomForestClassifier(n_estimators=config['random_forest']['n_estimators'], 
                                max_depth=config['random_forest']['max_depth'], 
                                random_state=config['random_forest']['random_state'])
    pipeline = create_pipeline(rf, use_pca)
    pipeline.fit(np.array(embedding_list).reshape(-1, EMBEDDING_SIZE), y)
    return pipeline

# Train models
classifiers = {}
for clf in config['classifiers']:
    if clf['name'].startswith('best'):
        continue  # Skip best models, we'll load them separately
    if clf['name'] == 'svm':
        classifiers[f"{clf['name']}{'_pca' if clf['use_pca'] else ''}"] = fit_svm(embedding_list, y, clf['use_pca'])
    elif clf['name'] == 'random_forest':
        classifiers[f"{clf['name']}{'_pca' if clf['use_pca'] else ''}"] = fit_rf(embedding_list, y, clf['use_pca'])

# Save models if configured
if config['training']['save_model']:
    for name, clf in classifiers.items():
        joblib.dump(clf, os.path.join(config['paths']['model_save_dir'], f'{name}_model_new.pkl'))

# Load best models
def load_best_models():
    models = {}
    for clf in config['classifiers']:
        if clf['name'].startswith('best'):
            models[clf['name']] = joblib.load(clf['model_path'])
    return models

best_models = load_best_models()

# Prepare test data
input_files = []
for folder in config['evaluation']['test_folders']:
    input_files.extend(glob.glob(f"{config['paths']['test_folder']}/{folder}/*.png"))

# Make predictions
predictions_df = pd.DataFrame(columns=['image_file'] + list(classifiers.keys()) + list(best_models.keys()))

for input_file in tqdm(input_files, desc="Making predictions"):
    new_image = load_image(input_file)
    with torch.no_grad():
        new_embedding = dinov2_vits14(new_image.to(device))
        new_embedding_array = np.array(new_embedding[0].cpu()).reshape(1, -1)
        
        row = {'image_file': input_file}
        
        # Predictions for trained models
        for name, clf in classifiers.items():
            row[name] = clf.predict(new_embedding_array)[0]
        
        # Predictions for best models
        for name, model in best_models.items():
            row[name] = model.predict(new_embedding_array)[0]
        
        predictions_df = predictions_df.append(row, ignore_index=True)

# Save predictions
predictions_df.to_csv(config['output']['predictions_file'], index=False)

# Compute and print metrics
predictions_df['true_label'] = predictions_df['image_file'].apply(lambda x: 'bull' if 'bull' in x else 'bear')

print("Classification Results:")
print("-----------------------")

for column in predictions_df.columns:
    if column not in ['image_file', 'true_label']:
        accuracy = accuracy_score(predictions_df['true_label'], predictions_df[column])
        conf_matrix = confusion_matrix(predictions_df['true_label'], predictions_df[column])
        
        print(f"{column}:")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Confusion Matrix:")
        print(conf_matrix)
        print()

print("Experiment completed successfully.")