import torch


def test_loop(dataloader, model, device):
    model.eval()
    num_batches = len(dataloader)

    test_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            images, _ = data
            images = images.to(device)

            extracted_feat, reconstructed_feat = model(images)
            loss = model.loss_fn(extracted_feat, reconstructed_feat).mean()

            test_loss = test_loss + loss.item()

    test_loss /= num_batches

    print(f"Test Epoch: \n Avg loss: {test_loss:>8f}\n")

    return test_loss
