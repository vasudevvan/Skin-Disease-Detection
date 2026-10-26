"""
Skin Disease Classification - MobileNetV2 Transfer Learning (3 classes)
Dataset: HAM10000 from Kaggle ("Skin Cancer MNIST: HAM10000")

Classes used:
    mel = Melanoma
    nv  = Melanocytic nevus (common mole)
    bkl = Benign keratosis

Expected folder layout (after unzipping the Kaggle download):
    ham10000/
        HAM10000_metadata.csv
        HAM10000_images_part_1/*.jpg
        HAM10000_images_part_2/*.jpg

Install (Colab already has these):
    pip install tensorflow pandas scikit-learn matplotlib seaborn

Run:
    python skin_disease_classifier.py

NOTE: Academic project only. Not a medical diagnosis tool.
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------
# 0. Settings (change these if needed)
# ----------------------------------------------------------------------
RAW_DIR = "ham10000"            # where you unzipped the Kaggle dataset
OUT_DIR = "skin_dataset"        # organised train/val/test folders will be created here
CLASSES = {"mel": "Melanoma", "nv": "Nevus", "bkl": "Benign_Keratosis"}
IMAGES_PER_CLASS = 1000         # more data per class improves accuracy (mel and bkl have ~1100 images)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15                     # initial training (base frozen)
FINE_TUNE_EPOCHS = 8             # second phase (gentle fine-tuning)
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ----------------------------------------------------------------------
# 1. Build a small, balanced dataset organised into folders
# ----------------------------------------------------------------------
def prepare_dataset():
    if os.path.exists(OUT_DIR):
        print(f"'{OUT_DIR}' already exists, skipping dataset preparation.")
        return

    # Works for Kaggle (HAM10000_metadata.csv) and Harvard Dataverse
    # (file named HAM10000_metadata, possibly tab-separated)
    meta_files = glob.glob(os.path.join(RAW_DIR, "**", "HAM10000_metadata*"), recursive=True)
    meta = pd.read_csv(meta_files[0], sep=None, engine="python")
    meta = meta[meta["dx"].isin(CLASSES.keys())]

    # map image_id -> file path (images are split across two folders)
    all_paths = glob.glob(os.path.join(RAW_DIR, "**", "*.jpg"), recursive=True)
    path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in all_paths}

    # take an equal number of images per class
    balanced = (
        meta.groupby("dx", group_keys=False)
        .apply(lambda g: g.sample(n=min(IMAGES_PER_CLASS, len(g)), random_state=SEED))
        .reset_index(drop=True)
    )
    print("Images per class:\n", balanced["dx"].value_counts(), "\n")

    # 70% train, 15% val, 15% test (stratified)
    train_df, temp_df = train_test_split(
        balanced, test_size=0.30, stratify=balanced["dx"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["dx"], random_state=SEED
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, row in split_df.iterrows():
            src = path_map.get(row["image_id"])
            if src is None:
                continue
            dst_dir = os.path.join(OUT_DIR, split_name, CLASSES[row["dx"]])
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, dst_dir)
        print(f"{split_name}: {len(split_df)} images")


def prepare_from_huggingface():
    """No Kaggle needed: downloads HAM10000 from Hugging Face (public, no account)."""
    if os.path.exists(OUT_DIR):
        print(f"'{OUT_DIR}' already exists, skipping dataset preparation.")
        return

    from datasets import load_dataset, concatenate_datasets

    print("Downloading HAM10000 from Hugging Face (about 3.7 GB, takes a few minutes)...")
    ds_dict = load_dataset("marmal88/skin_cancer")
    ds = concatenate_datasets([ds_dict[k] for k in ds_dict.keys()])

    def to_class(dx):
        v = str(dx).lower()
        if "nev" in v or v == "nv":
            return "Nevus"
        if v in ("mel", "melanoma"):
            return "Melanoma"
        if "keratosis" in v or v == "bkl":
            return "Benign_Keratosis"
        return None

    raw_labels = ds["dx"]
    print("Label values found in dataset:", sorted(set(raw_labels)))
    df = pd.DataFrame({"idx": range(len(ds)), "cls": [to_class(v) for v in raw_labels]})
    df = df.dropna()

    balanced = (
        df.groupby("cls", group_keys=False)
        .apply(lambda g: g.sample(n=min(IMAGES_PER_CLASS, len(g)), random_state=SEED))
        .reset_index(drop=True)
    )
    print("Images per class:\n", balanced["cls"].value_counts(), "\n")

    train_df, temp_df = train_test_split(
        balanced, test_size=0.30, stratify=balanced["cls"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["cls"], random_state=SEED
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, row in split_df.iterrows():
            dst_dir = os.path.join(OUT_DIR, split_name, row["cls"])
            os.makedirs(dst_dir, exist_ok=True)
            img = ds[int(row["idx"])]["image"].convert("RGB")
            img.save(os.path.join(dst_dir, f"{int(row['idx'])}.jpg"))
        print(f"{split_name}: {len(split_df)} images")


# Use a local folder if you downloaded the dataset yourself, otherwise get it
# automatically from Hugging Face (no Kaggle account required).
if os.path.exists(RAW_DIR):
    prepare_dataset()
else:
    prepare_from_huggingface()

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(OUT_DIR, "train"), image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, seed=SEED, shuffle=True,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(OUT_DIR, "val"), image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(OUT_DIR, "test"), image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=False,   # no shuffle so labels line up with predictions
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ----------------------------------------------------------------------
# 3. Build the model (MobileNetV2 + small classifier head)
# ----------------------------------------------------------------------
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False   # freeze pretrained layers first

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names))(x)   # logits
model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss_fn,
    metrics=["accuracy"],
)
model.summary()

# ----------------------------------------------------------------------
# 4. Phase 1: train only the new classifier head
# ----------------------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history1 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop]
)

# remember the best phase-1 model, so we can fall back to it if fine-tuning hurts
_, phase1_val_acc = model.evaluate(val_ds, verbose=0)
phase1_weights = model.get_weights()
print(f"Phase 1 validation accuracy: {phase1_val_acc:.3f}")

# ----------------------------------------------------------------------
# 5. Phase 2 (fine-tuning): unfreeze the last 30 layers, very small learning rate
# ----------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=loss_fn,
    metrics=["accuracy"],
)

early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=history1.epoch[-1] + 1 + FINE_TUNE_EPOCHS, initial_epoch=history1.epoch[-1] + 1,
    callbacks=[early_stop2],
)

# if fine-tuning made validation accuracy worse, go back to the phase-1 model
_, phase2_val_acc = model.evaluate(val_ds, verbose=0)
print(f"Phase 2 validation accuracy: {phase2_val_acc:.3f}")
if phase2_val_acc < phase1_val_acc:
    print("Fine-tuning did not help, restoring phase-1 weights.")
    model.set_weights(phase1_weights)

# ----------------------------------------------------------------------
# 6. Plot training curves
# ----------------------------------------------------------------------
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train")
plt.plot(val_acc, label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train")
plt.plot(val_loss, label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")

# ----------------------------------------------------------------------
# 7. Evaluate on the unseen test set
# ----------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.3f}")

y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# ----------------------------------------------------------------------
# 8. Show sample predictions
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 8))
for images, labels in test_ds.take(1):
    preds = np.argmax(model.predict(images), axis=1)
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}",
                  color=color, fontsize=9)
        plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")

# ----------------------------------------------------------------------
# 9. Save the model and define a helper to predict a single image
# ----------------------------------------------------------------------
model.save("skin_model.keras")
print("Model saved as skin_model.keras")


def predict_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)[np.newaxis, ...]
    probs = tf.nn.softmax(model.predict(arr)[0]).numpy()
    idx = int(np.argmax(probs))
    print(f"Prediction: {class_names[idx]} ({probs[idx] * 100:.1f}% confidence)")
    return class_names[idx]


# Example (uncomment and give a real image path):
# predict_image("skin_dataset/test/Melanoma/ISIC_0024310.jpg")

plt.show()
