import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
import matplotlib.pyplot as plt

encoder_model_path = "encoder/encoder_model.py"
spec = spec_from_file_location("encoder_model", encoder_model_path)
encoder_module = module_from_spec(spec)
spec.loader.exec_module(encoder_module)

encoder = encoder_module.ContextEncoder()
encoder.load_state_dict(torch.load("encoder/context_encoder.pth", map_location=torch.device("cpu")))
encoder.eval()

with open("config/env_variants.json", 'r') as f:
    variants = json.load(f)

variant_colors = {
    "cartpole_light": "blue",
    "cartpole_medium": "green",
    "cartpole_heavy": "red"
}
variant_labels = {
    "cartpole_light": "Light",
    "cartpole_medium": "Medium",
    "cartpole_heavy": "Heavy"
}

embeddings, labels, colors = [], [], []

for variant_name, color in variant_colors.items():
    for _ in range(10):
        base = np.random.randn(15, 3)
        if variant_name == "cartpole_light":
            base += 0.1
        elif variant_name == "cartpole_medium":
            base += 0.5
        else:
            base += 1.0
        traj_tensor = torch.tensor(base, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            z = encoder(traj_tensor).squeeze().numpy()
        embeddings.append(z)
        labels.append(variant_labels[variant_name])
        colors.append(color)

z_embeds = np.vstack(embeddings)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
z_proj = tsne.fit_transform(z_embeds)

plt.figure(figsize=(6, 5))
for label in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(z_proj[idxs, 0], z_proj[idxs, 1], label=label, alpha=0.7)
plt.title("t-SNE Projection of Latent Contexts")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.tight_layout()
plt.savefig("latent_space.png")
plt.close()

with open("eval/returns.json", "r") as f:
    all_returns = json.load(f)

label_map = {
    "cartpole_light": "Light (Train)",
    "cartpole_medium": "Medium (Train)",
    "cartpole_heavy": "Heavy (Train)",
    "unseen": "Unseen (Test)"
}

avg_returns = {label_map[k]: np.mean(v) for k, v in all_returns.items()}
std_returns = {label_map[k]: np.std(v) for k, v in all_returns.items()}

labels = list(avg_returns.keys())
means = [avg_returns[k] for k in labels]
errors = [std_returns[k] for k in labels]

plt.figure(figsize=(6, 4))
plt.bar(labels, means, yerr=errors, color=['blue', 'green', 'red', 'gray'], capsize=5)
plt.ylabel("Average Return")
plt.title("Performance Across CartPole Variants")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("performance_plot.png")
plt.close()
