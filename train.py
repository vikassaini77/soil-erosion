import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
import utils
import albumentations as A
import segmentation_models as sm
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
physical_devices = tf.config.list_physical_devices()
print(f"Available devices: {physical_devices}")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/plots", exist_ok=True)
print(f"Logs will be saved to: {log_dir}")

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

tf.config.optimizer.set_jit(True)

sm.set_framework('tf.keras')
sm.framework()

BASE_DIR = './processed_data/'
TRAIN_DIR = BASE_DIR + 'train/'
MASK_DIR = BASE_DIR + 'masks/'
INPUT_SIZE = (256,256,3)
BATCH_SIZE = 8
EPOCHS = 50

def load_data():
    train_files = sorted(os.listdir(TRAIN_DIR))
    mask_files = sorted(os.listdir(MASK_DIR))
    
    print(f"Train files: {len(train_files)}. ---> {train_files[:3]}")
    print(f"Test files: {len(mask_files)}. ---> {mask_files[:3]}")
    
    valid_pairs = []
    for train_file in train_files:
        base_name = os.path.splitext(train_file)[0]
        mask_file = base_name + '.png'
        if mask_file in mask_files:
            valid_pairs.append((train_file, mask_file))
    
    print(f"Valid image-mask pairs: {len(valid_pairs)}")
    
    out_rgb = []
    out_mask = []
    counter_empty = 0

    with tqdm(total=len(valid_pairs), desc="Loading data") as pbar:
        for p_img, p_mask in valid_pairs:
            img_path = os.path.join(TRAIN_DIR, p_img)
            mask_path = os.path.join(MASK_DIR, p_mask)

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.
            mask = cv2.imread(mask_path)

            # Resize if needed
            if img.shape[:2] != (256, 256):
                img = cv2.resize(img, (256, 256))
            if mask.shape[:2] != (256, 256):
                mask = cv2.resize(mask, (256, 256))

            mask = mask[:, :, :1]
            mask[mask > 0.] = 1.

            if 1 not in mask: counter_empty += 1

            if not (1 not in mask and counter_empty >= 40):
                out_rgb.append(img)
                out_mask.append(mask)
            
            pbar.update(1)

    return np.array(out_rgb, dtype='float32'), np.array(out_mask, dtype='float32')

# Visualize sample data
def visualize_samples(images, masks, num_samples=5, save_path=None):
    plt.figure(figsize=(15, 5*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Display original image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title(f"Image {i}")
        plt.axis('off')
        
        # Display mask
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title(f"Mask {i}")
        plt.axis('off')
        
        # Display overlay
        plt.subplot(num_samples, 3, i*3 + 3)
        overlay = images[i].copy()
        mask_rgb = np.concatenate([masks[i]] * 3, axis=2)
        overlay = overlay * 0.7 + mask_rgb * np.array([0, 1, 0]) * 0.3
        plt.imshow(overlay)
        plt.title(f"Overlay {i}")
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Sample visualization saved to {save_path}")
    plt.show()

# Load and split data
print("Loading dataset...")
out_rgb, out_mask = load_data()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    out_rgb, out_mask, test_size=0.05, shuffle=True, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")

# Visualize some samples
visualize_samples(X_train, y_train, num_samples=3, save_path=f"{log_dir}/plots/training_samples.png")

# Define augmentation
aug = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
    ],p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)])

# Optimize data loading for CPU
def generate_batches(X, y, batch_size):
    """Generate batches without using a generator function (more efficient)"""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_X = []
        batch_y = []
        
        for j in batch_indices:
            augmented = aug(image=X[j], mask=y[j])
            batch_X.append(augmented['image'])
            batch_y.append(augmented['mask'])
        
        batches.append((np.array(batch_X), np.array(batch_y)))
    
    return batches

# Define TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    update_freq='epoch',
    profile_batch=0
)

# Build model
model = sm.Unet(
    'efficientnetb0', 
    classes=1, 
    input_shape=(256, 256, 3),
    activation='sigmoid', 
    encoder_weights='imagenet'
)

print(f"Model summary:")
model.summary()

# Use legacy optimizer for M1/M2 Macs
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



# Compile model
model.compile(
    optimizer=optimizer, 
    loss=utils.FocalLoss, 
    metrics=[utils.dice_coef, 'binary_accuracy']
)

# Plot learning metrics history
def plot_learning_curves(history, save_path=None):
    metrics = list(history.keys())
    epochs = range(1, len(history[metrics[0]]) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Dice coefficient
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['dice_coef'], 'bo-', label='Training Dice')
    plt.plot(epochs, history['val_dice_coef'], 'ro-', label='Validation Dice')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    if 'binary_accuracy' in history:
        plt.plot(epochs, history['binary_accuracy'], 'bo-', label='Training Accuracy')
    if 'val_binary_accuracy' in history:
        plt.plot(epochs, history['val_binary_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot learning rate
    if 'lr' in history:
        plt.subplot(2, 2, 4)
        plt.semilogy(epochs, history['lr'], 'go-')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    plt.show()

print("Starting training...")
val_data = (X_test, y_test)

history = {
    'loss': [], 'dice_coef': [], 'binary_accuracy': [],
    'val_loss': [], 'val_dice_coef': [], 'val_binary_accuracy': [],
    'lr': []
}

best_val_loss = float('inf')
no_improvement_count = 0

tensorboard_callback.set_model(model)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Generate batches for this epoch
    batches = generate_batches(X_train, y_train, BATCH_SIZE)
    steps_per_epoch = min(200, len(batches))
    
    epoch_loss = []
    epoch_dice = []
    epoch_accuracy = []
    
    with tqdm(total=steps_per_epoch, desc=f"Training", unit="batch") as pbar:
        for i, (batch_X, batch_y) in enumerate(batches):
            if i >= steps_per_epoch:
                break
                
            # Train on batch
            metrics = model.train_on_batch(batch_X, batch_y)
            epoch_loss.append(metrics[0])
            epoch_dice.append(metrics[1])
            epoch_accuracy.append(metrics[2])
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f"{metrics[0]:.4f}", 
                'dice': f"{metrics[1]:.4f}", 
                'acc': f"{metrics[2]:.4f}"
            })
            pbar.update(1)
    
    val_loss = []
    val_dice = []
    val_acc = []
    
    val_batch_size = 16
    num_val_batches = int(np.ceil(len(X_test) / val_batch_size))
    
    with tqdm(total=num_val_batches, desc="Validation", unit="batch") as pbar:
        for i in range(num_val_batches):
            start_idx = i * val_batch_size
            end_idx = min((i + 1) * val_batch_size, len(X_test))
            
            val_batch_X = X_test[start_idx:end_idx]
            val_batch_y = y_test[start_idx:end_idx]
            
            metrics = model.evaluate(val_batch_X, val_batch_y, verbose=0)
            val_loss.append(metrics[0])
            val_dice.append(metrics[1])
            val_acc.append(metrics[2])
            
            pbar.update(1)
    
    avg_loss = np.mean(epoch_loss)
    avg_dice = np.mean(epoch_dice)
    avg_accuracy = np.mean(epoch_accuracy)
    
    avg_val_loss = np.mean(val_loss)
    avg_val_dice = np.mean(val_dice)
    avg_val_acc = np.mean(val_acc)
    
    history['loss'].append(avg_loss)
    history['dice_coef'].append(avg_dice)
    history['binary_accuracy'].append(avg_accuracy)
    history['val_loss'].append(avg_val_loss)
    history['val_dice_coef'].append(avg_val_dice)
    history['val_binary_accuracy'].append(avg_val_acc)
    history['lr'].append(float(K.get_value(model.optimizer.learning_rate)))
    
    logs = {
        'loss': avg_loss,
        'dice_coef': avg_dice,
        'binary_accuracy': avg_accuracy,
        'val_loss': avg_val_loss,
        'val_dice_coef': avg_val_dice,
        'val_binary_accuracy': avg_val_acc,
        'lr': float(K.get_value(model.optimizer.learning_rate))
    }
    tensorboard_callback.on_epoch_end(epoch, logs)
    
    # Print epoch summary
    print(f"\n{'='*30} Epoch {epoch+1}/{EPOCHS} Summary {'='*30}")
    print(f"Loss: {avg_loss:.4f} - Dice: {avg_dice:.4f} - Accuracy: {avg_accuracy:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f} - Val Dice: {avg_val_dice:.4f} - Val Accuracy: {avg_val_acc:.4f}")
    print(f"Learning Rate: {float(K.get_value(model.optimizer.learning_rate)):.6f}")
    
    # Plot the current learning curves every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        plot_learning_curves(history, save_path=f"{log_dir}/plots/learning_curves_epoch_{epoch+1}.png")
    
    # Handle model checkpointing manually
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"✅ New best model with val_loss: {best_val_loss:.4f}")
        model.save('crusa.keras')
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        print(f"⚠️ No improvement for {no_improvement_count} epochs. Best val_loss: {best_val_loss:.4f}")
        
    # Early stopping check
    if no_improvement_count >= 8:
        print(f"🛑 Early stopping triggered after {epoch+1} epochs")
        break
        
    # Learning rate scheduling - halve LR after 3 epochs without improvement
    if no_improvement_count > 0 and no_improvement_count % 3 == 0:
        current_lr = float(K.get_value(model.optimizer.learning_rate))
        new_lr = current_lr * 0.5
        if new_lr >= 1e-6:
            print(f"📉 Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
            K.set_value(model.optimizer.learning_rate, new_lr)

tensorboard_callback.on_train_end(None)

plot_learning_curves(history, save_path=f"{log_dir}/plots/final_learning_curves.png")

print("Generating final predictions for evaluation...")
y_pred = model.predict(X_test, verbose=1)
y_pred_binary = (y_pred > 0.5).astype(np.float32)

y_true_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
y_pred_binary_flat = y_pred_binary.flatten()

cm = confusion_matrix(y_true_flat, y_pred_binary_flat)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(f"{log_dir}/plots/confusion_matrix.png")
plt.show()

fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(f"{log_dir}/plots/roc_curve.png")
plt.show()

precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(f"{log_dir}/plots/precision_recall_curve.png")
plt.show()

def visualize_predictions(images, true_masks, pred_masks, num_samples=5, save_path=None):
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    plt.figure(figsize=(15, 5*len(indices)))
    
    for i, idx in enumerate(indices):
        # Display original image
        plt.subplot(len(indices), 4, i*4 + 1)
        plt.imshow(images[idx])
        plt.title(f"Image")
        plt.axis('off')
        
        # Display true mask
        plt.subplot(len(indices), 4, i*4 + 2)
        plt.imshow(true_masks[idx].squeeze(), cmap='gray')
        plt.title(f"True Mask")
        plt.axis('off')
        
        # Display predicted mask
        plt.subplot(len(indices), 4, i*4 + 3)
        plt.imshow(pred_masks[idx].squeeze(), cmap='gray')
        plt.title(f"Predicted Mask")
        plt.axis('off')
        
        # Display overlay
        plt.subplot(len(indices), 4, i*4 + 4)
        overlay = images[idx].copy()
        # Green: True positive, Red: False positive, Blue: False negative
        true_mask_rgb = true_masks[idx].squeeze()
        pred_mask_rgb = pred_masks[idx].squeeze()
        
        mask_vis = np.zeros((*true_mask_rgb.shape, 3))
        mask_vis[..., 1] = (true_mask_rgb * pred_mask_rgb)  # True positive - green
        mask_vis[..., 0] = (pred_mask_rgb * (1 - true_mask_rgb))  # False positive - red
        mask_vis[..., 2] = (true_mask_rgb * (1 - pred_mask_rgb))  # False negative - blue
        
        plt.imshow(images[idx] * 0.7 + mask_vis * 0.3)
        plt.title("Overlay (Green: TP, Red: FP, Blue: FN)")
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    plt.show()

visualize_predictions(X_test, y_test, y_pred_binary, num_samples=5, save_path=f"{log_dir}/plots/prediction_examples.png")

model.save('model.keras')
print(f"Model saved as model.keras, logs and visualizations saved in {log_dir}")