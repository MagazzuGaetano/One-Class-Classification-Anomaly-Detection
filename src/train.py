def train_loop(dataloader, model, optimizer, device, train_print_freq):
    num_batches = len(dataloader)
    model.train()

    train_loss = 0.0
    for batch, data in enumerate(dataloader):
        images, _ = data
        images = images.to(device)

        extracted_feat, reconstructed_feat = model(images)
        loss = model.loss_fn(extracted_feat, reconstructed_feat).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss = train_loss + loss.item()

        if batch % train_print_freq == 0:
            print(
                f"[{batch:>5d}/{num_batches:>5d}] loss: {loss.item():>7f}"
            )

    train_loss /= num_batches

    print(f"Train Epoch: \n Avg loss: {loss:>7f}")

    return train_loss
