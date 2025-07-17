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


# Create and train the model
model = build_model()
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# Evaluate the model on clean test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy on clean images: {test_acc:.4f}")


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


# Generate adversarial examples using PGD
# Select 100 test samples for demonstration
num_samples = 100
x_test_subset = x_test[:num_samples]
y_test_subset = y_test[:num_samples]

# PGD attack parameters
epsilon = 0.1  # Maximum perturbation
alpha = 0.01  # Step size
num_iter = 20  # Number of iterations

# Generate adversarial examples
adv_examples = pgd_attack(model, x_test_subset, y_test_subset, epsilon, alpha, num_iter)

# Evaluate the model on adversarial examples
adv_predictions = model.predict(adv_examples)
adv_accuracy = np.mean(np.argmax(adv_predictions, axis=1) == np.argmax(y_test_subset, axis=1))
print(f"Test accuracy on adversarial examples: {adv_accuracy:.4f}")


# Function to visualize original and adversarial examples
def visualize_examples(clean_images, adv_images, clean_labels, model):
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
    plt.savefig('pgd_adversarial_examples.png')
    plt.show()


# Visualize the results
visualize_examples(x_test_subset, adv_examples, y_test_subset, model)

# Calculate and print perturbation statistics
perturbation = adv_examples - x_test_subset
mean_perturbation = np.mean(np.abs(perturbation))
max_perturbation = np.max(np.abs(perturbation))

print(f"Mean absolute perturbation: {mean_perturbation:.6f}")
print(f"Maximum perturbation (L-infinity norm): {max_perturbation:.6f}")