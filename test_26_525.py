import argparse
import contextlib
import io
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from MIDataSet import MultimodalRegistrationDataset
from module.UTN_More_Layers_Init_Lambda import UTNet
from module.utils import label_to_one_hot, label_value_to_index, warp_images_grid_sample


def dice_per_sample(pred, target, eps=1e-6):
    pred_nonzero = pred > 0
    target_nonzero = target > 0
    intersection = ((pred == target) & (pred_nonzero & target_nonzero)).float().sum(dim=(1, 2, 3))
    a = pred_nonzero.float().sum(dim=(1, 2, 3))
    b = target_nonzero.float().sum(dim=(1, 2, 3))
    return (2 * intersection + eps) / (a + b + eps)


def make_dataset(root, split):
    split_dir = os.path.join(root, split)
    transform = transforms.Compose([transforms.ToTensor()])
    return MultimodalRegistrationDataset(
        os.path.join(split_dir, "t1_warp"),
        os.path.join(split_dir, "t2"),
        os.path.join(split_dir, "seg"),
        os.path.join(split_dir, "seg_warp"),
        os.path.join(split_dir, "t1"),
        transform=transform,
    )


def make_model(device, batch_size, num_layers):
    model = UTNet(
        beta=0.01,
        enc_nf=[16, 32, 32, 32],
        dec_nf=[32, 32, 32, 32, 32, 16, 16],
        size=[256, 256],
        device=device,
        size_tensor=(batch_size, 1, 256, 256),
        num_layers=num_layers,
        shold_values=0.16,
    )
    return model.to(device)


def layer_segmentations(model, model1_img, model2_img, seg):
    with contextlib.redirect_stdout(io.StringIO()):
        _, flow = model.pre_reg(model1_img, model2_img)

    fai_result1, fai_result2 = torch.split(flow / 255.0, 1, dim=1)
    model1_init = warp_images_grid_sample(model1_img, fai_result1 * 255.0, fai_result2 * 255.0)
    seg_idx = label_value_to_index(seg * 255.0)
    seg_1h = label_to_one_hot(seg_idx, num_classes=4)
    mapping = torch.tensor([0, 64, 128, 255], device=seg.device)

    outputs = []
    for tnrd_layer in model.tnrd_layers:
        model_input = torch.cat((model1_init, model2_img), dim=1)
        with contextlib.redirect_stdout(io.StringIO()):
            fai = model.unet(model_input)
        fai_unet_1, fai_unet_2 = torch.split(fai, 1, dim=1)
        fai_result_1, fai_result_2 = tnrd_layer(fai_result1, fai_result2, fai_unet_1, fai_unet_2)
        fai_result = torch.cat((fai_result_1, fai_result_2), dim=1)
        model1_init = warp_images_grid_sample(model1_img, fai_result_1 * 255.0, fai_result_2 * 255.0)
        fai_result1, fai_result2 = torch.split(fai_result, 1, dim=1)

        seg_wrapped = warp_images_grid_sample(seg_1h, fai_result1 * 255.0, fai_result2 * 255.0)
        seg_discrete = mapping[torch.argmax(seg_wrapped, dim=1)].unsqueeze(1)
        outputs.append(seg_discrete)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate 26_525 T=7 elasticity model on data8k10.")
    parser.add_argument("--data-root", default="data8k10")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--weights", default=os.path.join("26_525", "best_dice_model_swin.pth"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--layer-dice", action="store_true", help="Report Dice after each unfolded layer.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_dataset(args.data_root, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(device, args.batch_size, args.num_layers)
    state = torch.load(args.weights, map_location=device)
    load_info = model.load_state_dict(state, strict=True)
    model.eval()

    final_dice = []
    layer_dice = [[] for _ in range(args.num_layers)]
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            model1_img = batch["model1"].to(device)
            model2_img = batch["model2"].to(device)
            seg = batch["seg"].to(device)
            label = batch["label"].to(device) * 255.0

            if args.layer_dice:
                segs = layer_segmentations(model, model1_img, model2_img, seg)
                for idx, pred in enumerate(segs):
                    layer_dice[idx].append(dice_per_sample(pred, label).cpu())
                seg_wrapped = segs[-1]
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, seg_wrapped = model(model1_img, model2_img, seg)

            final_dice.append(dice_per_sample(seg_wrapped, label).cpu())

    final_values = torch.cat(final_dice)
    print(f"load_info: {load_info}")
    print(f"device: {device}")
    print(f"samples: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"final_dice_mean: {final_values.mean().item():.8f}")
    print(f"final_dice_std: {final_values.std(unbiased=False).item():.8f}")
    print(f"elapsed_sec: {time.time() - start:.2f}")

    if args.layer_dice:
        for idx, values in enumerate(layer_dice, start=1):
            values = torch.cat(values)
            print(f"layer_{idx}_dice_mean: {values.mean().item():.8f}")


if __name__ == "__main__":
    main()
