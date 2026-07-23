import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one complete epoch.
    Returns the average loss and accuracy.
    """

    # Put the model in training mode.
    # This enables layers like Dropout.
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Process one batch at a time.
    for images, labels in dataloader:

        # Move the batch to the selected device (CPU or GPU).
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous iteration.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate prediction error.
        loss = criterion(outputs, labels)

        # Compute gradients.
        loss.backward()

        # Update model weights.
        optimizer.step()

        # Store loss for averaging later.
        running_loss += loss.item()

        # Get predicted emotion for each image.
        _, predictions = torch.max(outputs, dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = running_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples

    return average_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """

    # Switch to evaluation mode.
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Gradients are not needed during validation.
    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predictions = torch.max(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = running_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples

    return average_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    checkpoint_path
):
    """
    Runs the complete training process.
    """

    best_validation_accuracy = 0

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Save the model only if validation improves.
        if val_accuracy > best_validation_accuracy:

            best_validation_accuracy = val_accuracy

            torch.save(model.state_dict(), checkpoint_path)

            print("Best model saved.")