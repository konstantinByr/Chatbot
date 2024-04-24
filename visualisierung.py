import torch
from model import NeuralNet

# Lade trainiertes Modell aus der gespeicherten Datei
checkpoint = torch.load('data.pth')
inputSize = checkpoint['inputSize']
hiddenSize = checkpoint['hiddenSize']
outputSize = checkpoint['outputSize']
allWords = checkpoint['allWords']
tags = checkpoint['tags']
model_state = checkpoint['model_state']

# Erstelle eine Instanz des Modells
model = NeuralNet(inputSize, hiddenSize, outputSize)
model.load_state_dict(model_state)
model.eval()  # Setze das Modell in den Evaluierungsmodus

# Erzeuge Dummy-Daten
batch_size = 32
dummy_input = torch.randn(batch_size, inputSize)

# Führe den Vorwärtsdurchlauf durch und speichere die Aktivierungen
with torch.no_grad():
    activations = []
    for i in range(len(model.activations)):
        activations.append([])  # Initialisiere eine leere Liste für jede Schicht
    for batch in dummy_input:
        batch = batch.unsqueeze(0)  # Füge eine zusätzliche Dimension hinzu (Batch-Dimension)
        output = model(batch)
        for i, activation in enumerate(model.activations):
            activations[i].append(activation.clone().detach())

# Überprüfe die Struktur der activations-Liste
for i, layer_activations in enumerate(activations):
    print(f"Layer {i+1} activations: {len(layer_activations)} tensors")
    for j, tensor in enumerate(layer_activations):
        print(f"  Tensor {j+1}: {tensor.shape}")



# Visualisiere die Aktivierungen
def visualize_activations(activations, max_neurons_per_plot=10):
    num_layers = len(activations)
    for i in range(num_layers):
        layer_activations = torch.cat(activations[i]).cpu().detach().numpy()
        num_neurons = layer_activations.shape[1]
        
        num_plots = int(np.ceil(num_neurons / max_neurons_per_plot))
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots), sharex=True)

        for plot_index in range(num_plots):
            start_neuron = plot_index * max_neurons_per_plot
            end_neuron = min((plot_index + 1) * max_neurons_per_plot, num_neurons)

            ax = axes[plot_index] if num_plots > 1 else axes
            for j in range(start_neuron, end_neuron):
                ax.plot(layer_activations[:, j], label=f'Neuron {j+1}')
            ax.set_title(f'Layer {i+1} Activations ({start_neuron+1}-{end_neuron})')
            ax.set_ylabel('Activation Value')
            ax.grid(True)

            if plot_index == num_plots - 1:
                ax.set_xlabel('Samples')

        plt.tight_layout()
        plt.show()