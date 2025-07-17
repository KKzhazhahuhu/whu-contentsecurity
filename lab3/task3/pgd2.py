import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import idx2numpy

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST dataset
x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
x_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape images to (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Build a simple CNN model for MNIST
def build_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# PGD Attack implementation
def pgd_attack(model, x, y, epsilon, alpha, num_iter):
    """
    Projected Gradient Descent (PGD) attack

    Args:
        model: The target model to attack
        x: Clean input images
        y: True labels for the images
        epsilon: Maximum perturbation size (L-infinity norm)
        alpha: Step size for each iteration
        num_iter: Number of iterations

    Returns:
        Adversarial examples
    """
    # Create a TF constant for the target labels
    y_tf = tf.constant(y, dtype=tf.float32)

    # Initialize adversarial examples with small random noise
    x_adv = x + np.random.uniform(-epsilon, epsilon, x.shape).astype('float32')
    x_adv = np.clip(x_adv, 0.0, 1.0)

    for i in range(num_iter):
        # Convert to TensorFlow tensor
        x_adv_tf = tf.Variable(x_adv)

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tf)
            predictions = model(x_adv_tf)
            loss = tf.keras.losses.categorical_crossentropy(y_tf, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, x_adv_tf)

        # Update adversarial examples using the sign of gradients
        x_adv = x_adv + alpha * np.sign(gradients.numpy())

        # Project back to epsilon ball
        x_adv = np.clip(x_adv, x - epsilon, x + epsilon)

        # Ensure valid pixel values
        x_adv = np.clip(x_adv, 0.0, 1.0)

    return x_adv


# Function to visualize original and adversarial examples
def visualize_examples(clean_images, adv_images, clean_labels, model, title="PGD Adversarial Examples"):
    n = 5  # Number of examples to show
    plt.figure(figsize=(15, 3 * n))

    for i in range(n):
        # Original image
        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(clean_images[i].reshape(28, 28), cmap='gray')
        clean_pred = np.argmax(model.predict(clean_images[i:i + 1])[0])
        clean_true = np.argmax(clean_labels[i])
        plt.title(f"Original: True {clean_true}, Pred {clean_pred}")
        plt.axis('off')

        # Adversarial image
        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(adv_images[i].reshape(28, 28), cmap='gray')
        adv_pred = np.argmax(model.predict(adv_images[i:i + 1])[0])
        plt.title(f"Adversarial: True {clean_true}, Pred {adv_pred}")
        plt.axis('off')

        # Perturbation (difference)
        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(np.abs(adv_images[i] - clean_images[i]).reshape(28, 28), cmap='viridis')
        plt.title(f"Perturbation (magnified)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()


# Function to evaluate model performance on clean and adversarial examples
def evaluate_model(model, x_clean, y_clean, x_adv=None, y_adv=None):
    # Evaluate on clean examples
    clean_loss, clean_acc = model.evaluate(x_clean, y_clean, verbose=0)
    print(f"Test accuracy on clean examples: {clean_acc:.4f}")

    # Evaluate on adversarial examples if provided
    if x_adv is not None and y_adv is not None:
        adv_loss, adv_acc = model.evaluate(x_adv, y_adv, verbose=0)
        print(f"Test accuracy on adversarial examples: {adv_acc:.4f}")
        return clean_acc, adv_acc

    return clean_acc, None


# === Step 1: Train a baseline model ===
print("=== Training Baseline Model ===")
baseline_model = build_model()
baseline_history = baseline_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# === Step 2: Generate adversarial examples using PGD ===
print("\n=== Generating Adversarial Examples ===")
# PGD attack parameters
epsilon = 0.1  # Maximum perturbation
alpha = 0.01  # Step size
num_iter = 20  # Number of iterations

# Generate adversarial examples for a subset of the test set
num_test_samples = 1000  # Use 1000 test samples for evaluation
x_test_subset = x_test[:num_test_samples]
y_test_subset = y_test[:num_test_samples]

# Generate adversarial examples for evaluation
test_adv_examples = pgd_attack(baseline_model, x_test_subset, y_test_subset, epsilon, alpha, num_iter)

# Visualize some examples
visualize_examples(x_test_subset, test_adv_examples, y_test_subset, baseline_model, "Baseline Model PGD Examples")

# Evaluate baseline model performance
print("\n=== Evaluating Baseline Model ===")
baseline_clean_acc, baseline_adv_acc = evaluate_model(baseline_model, x_test_subset, y_test_subset, test_adv_examples,
                                                      y_test_subset)

# === Step 3: Generate adversarial examples for training ===
print("\n=== Generating Adversarial Examples for Training ===")
# We'll use a subset of the training data to generate adversarial examples
# Let's try different sizes of adversarial examples for training

# Define the proportions of adversarial examples to include in training
adv_proportions = [0.1, 0.2, 0.3]
results = []  # To store results for each proportion

for adv_prop in adv_proportions:
    print(f"\n=== Training with {adv_prop * 100}% Adversarial Examples ===")

    # Number of training examples to use for generating adversarial examples
    num_adv_train = int(adv_prop * len(x_train))

    # Subset of training data for generating adversarial examples
    x_train_subset = x_train[:num_adv_train]
    y_train_subset = y_train[:num_adv_train]

    # Generate adversarial examples for training
    print(f"Generating {num_adv_train} adversarial examples for training...")
    train_adv_examples = pgd_attack(baseline_model, x_train_subset, y_train_subset, epsilon, alpha, num_iter)

    # === Step 4: Train model with mixture of clean and adversarial examples ===
    # Create augmented training dataset with both clean and adversarial examples
    x_train_augmented = np.concatenate([x_train, train_adv_examples], axis=0)
    y_train_augmented = np.concatenate([y_train, y_train_subset], axis=0)

    # Shuffle the combined data
    indices = np.random.permutation(len(x_train_augmented))
    x_train_augmented = x_train_augmented[indices]
    y_train_augmented = y_train_augmented[indices]

    # Create and train a new model with the augmented dataset
    robust_model = build_model()
    robust_history = robust_model.fit(
        x_train_augmented, y_train_augmented,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # === Step 5: Evaluate robust model on clean and adversarial examples ===
    print(f"\n=== Evaluating Robust Model ({adv_prop * 100}% Adversarial Examples) ===")

    # Generate new adversarial examples based on the robust model
    robust_test_adv = pgd_attack(robust_model, x_test_subset, y_test_subset, epsilon, alpha, num_iter)

    # Visualize examples with the robust model
    visualize_examples(
        x_test_subset, robust_test_adv, y_test_subset, robust_model,
        f"Robust Model ({adv_prop * 100}% Adv) PGD Examples"
    )

    # Evaluate performance
    robust_clean_acc, _ = evaluate_model(robust_model, x_test_subset, y_test_subset)

    _, robust_on_baseline_adv_acc = evaluate_model(
        robust_model,
        x_test_subset,  # 干净样本
        y_test_subset,
        test_adv_examples,  # 对抗样本
        y_test_subset  # 对抗样本标签
    )

    _, robust_on_new_adv_acc = evaluate_model(
        robust_model,
        x_test_subset,  # 干净样本
        y_test_subset,
        robust_test_adv,  # 对抗样本
        y_test_subset  # 对抗样本标签
    )

# === Step 6: Summarize and visualize results ===
print("\n=== Summary of Results ===")
print(f"Baseline Model - Clean Accuracy: {baseline_clean_acc:.4f}, Adversarial Accuracy: {baseline_adv_acc:.4f}")

for result in results:
    print(f"\nRobust Model ({result['adv_proportion'] * 100}% Adversarial Examples):")
    print(f"  Clean Accuracy: {result['clean_acc']:.4f}")
    print(f"  Accuracy on Baseline Adversarial Examples: {result['baseline_adv_acc']:.4f}")
    print(f"  Accuracy on New Adversarial Examples: {result['new_adv_acc']:.4f}")

# Plot results
plt.figure(figsize=(12, 8))

# Plot accuracy on clean examples
plt.subplot(1, 2, 1)
plt.plot([0] + [r['adv_proportion'] for r in results], [baseline_clean_acc] + [r['clean_acc'] for r in results], 'o-',
         label='Clean Accuracy')
plt.xlabel('Proportion of Adversarial Examples in Training')
plt.ylabel('Accuracy')
plt.title('Effect on Clean Examples')
plt.grid(True)
plt.legend()

# Plot accuracy on adversarial examples
plt.subplot(1, 2, 2)
plt.plot([0] + [r['adv_proportion'] for r in results], [baseline_adv_acc] + [r['baseline_adv_acc'] for r in results],
         'o-', label='Accuracy on Baseline Adv')
plt.plot([0] + [r['adv_proportion'] for r in results], [baseline_adv_acc] + [r['new_adv_acc'] for r in results], 's-',
         label='Accuracy on New Adv')
plt.xlabel('Proportion of Adversarial Examples in Training')
plt.ylabel('Accuracy')
plt.title('Effect on Adversarial Examples')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('adversarial_training_results.png')
plt.show()


# === Step 7: Additional Analysis: Impact on decision boundaries ===
# Let's look at the confidence of predictions

def analyze_prediction_confidence(model, x_clean, x_adv, y_true):
    """Analyze prediction confidence on clean and adversarial examples"""
    # Get predictions
    clean_preds = model.predict(x_clean)
    adv_preds = model.predict(x_adv)

    # Calculate max confidence for each prediction
    clean_confidence = np.max(clean_preds, axis=1)
    adv_confidence = np.max(adv_preds, axis=1)

    # Get predicted classes
    clean_pred_classes = np.argmax(clean_preds, axis=1)
    adv_pred_classes = np.argmax(adv_preds, axis=1)
    true_classes = np.argmax(y_true, axis=1)

    # Calculate correct and incorrect predictions
    clean_correct = clean_pred_classes == true_classes
    adv_correct = adv_pred_classes == true_classes

    return {
        'clean_confidence': clean_confidence,
        'adv_confidence': adv_confidence,
        'clean_correct': clean_correct,
        'adv_correct': adv_correct
    }


# Analyze baseline model
baseline_confidence = analyze_prediction_confidence(
    baseline_model, x_test_subset, test_adv_examples, y_test_subset
)

# Analyze robust model (using the last one we trained)
robust_model = build_model()  # The last one we trained
robust_confidence = analyze_prediction_confidence(
    robust_model, x_test_subset, robust_test_adv, y_test_subset
)

# Plot confidence histograms
plt.figure(figsize=(15, 10))

# Baseline model confidence on clean examples
plt.subplot(2, 2, 1)
plt.hist(baseline_confidence['clean_confidence'], bins=20, alpha=0.7, color='blue')
plt.title('Baseline Model: Confidence on Clean Examples')
plt.xlabel('Confidence')
plt.ylabel('Frequency')

# Baseline model confidence on adversarial examples
plt.subplot(2, 2, 2)
plt.hist(baseline_confidence['adv_confidence'], bins=20, alpha=0.7, color='red')
plt.title('Baseline Model: Confidence on Adversarial Examples')
plt.xlabel('Confidence')
plt.ylabel('Frequency')

# Robust model confidence on clean examples
plt.subplot(2, 2, 3)
plt.hist(robust_confidence['clean_confidence'], bins=20, alpha=0.7, color='green')
plt.title('Robust Model: Confidence on Clean Examples')
plt.xlabel('Confidence')
plt.ylabel('Frequency')

# Robust model confidence on adversarial examples
plt.subplot(2, 2, 4)
plt.hist(robust_confidence['adv_confidence'], bins=20, alpha=0.7, color='orange')
plt.title('Robust Model: Confidence on Adversarial Examples')
plt.xlabel('Confidence')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('confidence_analysis.png')
plt.show()

# Calculate and display resilience score - ratio of adversarial to clean accuracy
baseline_resilience = baseline_adv_acc / baseline_clean_acc
print(f"\n=== Resilience Analysis ===")
print(f"Baseline Model Resilience Score: {baseline_resilience:.4f}")

for result in results:
    resilience_score = result['new_adv_acc'] / result['clean_acc']
    print(f"Robust Model ({result['adv_proportion'] * 100}% Adv) Resilience Score: {resilience_score:.4f}")
    print(f"Relative Improvement: {(resilience_score / baseline_resilience - 1) * 100:.2f}%")