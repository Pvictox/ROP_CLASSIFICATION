import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


#CLAUDE REINOU AQUI

def _get_target_layer(model):
    if hasattr(model, "features"):
        return [model.features[-1]]
    if hasattr(model, "conv_head"):
        return [model.conv_head]
    if hasattr(model, "blocks"):
        return [model.blocks[-1]]

    raise ValueError(
        "Não foi possível detectar a target_layer automaticamente. "
        "Passe `target_layer=[model.<camada>]` manualmente."
    )


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().numpy()

    # Se vier como (C, H, W) → (H, W, C)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)

    # Reverte normalização ImageNet (mean/std padrão)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = std * img + mean
    img  = np.clip(img, 0.0, 1.0).astype(np.float32)

    # Garante 3 canais (grayscale → RGB)
    if img.ndim == 2 or img.shape[-1] == 1:
        img = np.repeat(img.reshape(*img.shape[:2], 1), 3, axis=-1)

    return img


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def plot_gradcam(
    model,
    dataset,
    n_images: int = 12,
    class_names: list[str] | None = None,
    target_layer=None,
    cam_method=GradCAM,
    device: str | torch.device = "cpu",
    indices: list[int] | None = None,
    figsize_per_image: tuple[float, float] = (3.5, 4.2),
    ncols: int = 4,
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
    colormap: str = "jet",
    save_path: str | None = None,
    show: bool = True,
):
   
    # --- Setup ---
    device   = torch.device(device)
    model    = model.to(device).eval()

    if class_names is None:
        class_names = ["Classe 0", "Classe 1"]

    if target_layer is None:
        target_layer = _get_target_layer(model)

    # --- Seleção dos índices ---
    n_total = len(dataset)
    if indices is None:
        n_images = min(n_images, n_total)
        indices  = np.random.choice(n_total, size=n_images, replace=False).tolist()
    else:
        n_images = len(indices)

    # --- Layout da figura ---
    ncols = min(ncols, n_images)
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_image[0] * ncols, figsize_per_image[1] * nrows),
    )
    axes = np.array(axes).reshape(-1)   # garante array 1-D mesmo com 1 linha

    # --- Loop GradCAM ---
    with GradCAM(model=model, target_layers=target_layer) as cam:
        for plot_idx, sample_idx in enumerate(indices):
            ax = axes[plot_idx]

            # 1. Carrega amostra
            sample = dataset[sample_idx]
            tensor, true_label = sample[0], int(sample[1])

            input_tensor = tensor.unsqueeze(0).to(device)   # (1, C, H, W)
            rgb_img      = _tensor_to_rgb(tensor)            # (H, W, 3) float32

            # 2. Predição
            with torch.no_grad():
                logits = model(input_tensor)
                probs  = torch.softmax(logits, dim=1)[0]

            pred_class = int(probs.argmax().item())
            pred_prob  = float(probs[pred_class].item())

            # 3. GradCAM — explica a classe predita
            targets       = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(
                input_tensor  = input_tensor,
                targets       = targets,
                aug_smooth    = aug_smooth,
                eigen_smooth  = eigen_smooth,
            )[0]   # (H, W)

            # 4. Sobreposição
            visualization = show_cam_on_image(
                rgb_img, grayscale_cam, use_rgb=True, colormap=colormap
            )

            # 5. Plot
            ax.imshow(visualization)
            ax.axis("off")

            correct   = (pred_class == true_label)
            edge_color = "#2ecc71" if correct else "#e74c3c"   # verde / vermelho

            title = (
                f"Real: {class_names[true_label]}\n"
                f"Pred: {class_names[pred_class]}  ({pred_prob:.1%})"
            )
            ax.set_title(title, fontsize=8.5, pad=4)

            # Borda colorida indicando acerto/erro
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(3)

    # --- Desativa eixos extras ---
    for ax in axes[n_images:]:
        ax.axis("off")

    # --- Legenda global ---
    legend_handles = [
        mpatches.Patch(color="#2ecc71", label="Predição correta"),
        mpatches.Patch(color="#e74c3c", label="Predição incorreta"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle("GradCAM — Visualização de Ativações", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    if show:
        plt.show()

    return fig, axes


def plot_gradcam_split(
    model,
    dataset,
    n_per_group: int = 6,
    **kwargs,
):
    """
    Roda a inferência no dataset inteiro e plota dois grids separados:
    um para acertos e outro para erros.

    """
    device = torch.device(kwargs.get("device", "cpu"))
    model  = model.to(device).eval()

    correct_idx = []
    wrong_idx   = []

    print("Classificando dataset para separar acertos e erros...")
    with torch.no_grad():
        for i in range(len(dataset)):
            tensor, label = dataset[i][0], int(dataset[i][1])
            logits = model(tensor.unsqueeze(0).to(device))
            pred   = int(logits.argmax(dim=1).item())
            if pred == label:
                correct_idx.append(i)
            else:
                wrong_idx.append(i)

    print(f"  Acertos : {len(correct_idx)}  |  Erros : {len(wrong_idx)}")

    figs = {}

    if correct_idx:
        print("\n--- Acertos ---")
        figs["correct"] = plot_gradcam(
            model, dataset,
            indices=correct_idx[:n_per_group],
            **kwargs,
        )

    if wrong_idx:
        print("\n--- Erros ---")
        figs["wrong"] = plot_gradcam(
            model, dataset,
            indices=wrong_idx[:n_per_group],
            **kwargs,
        )

    return figs
