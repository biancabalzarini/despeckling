from torch.utils.data import Dataset
import torch.nn as nn
import torch


class InMemoryImageDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        assert len(input_images) == len(target_images), "Input and target image sets must have the same length"
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

def split_images(images, side_size):
    """
    Divide un array de imágenes en subimágenes más pequeñas.
    
    Parameters
    ----------
        images: np.array
            Array de forma (n_images, height, width)
        side_size: int
            Tamaño de las subimágenes (cuadradas)
    
    Returns
    -------
        Array de forma (n_images * n_subimages, side_size, side_size)
    """
    n_images, height, width = images.shape
    n_h = height // side_size
    n_w = width // side_size
    
    cropped = images[:, :n_h * side_size, :n_w * side_size]
    
    subimages = cropped.reshape(n_images, n_h, side_size, n_w, side_size)
    subimages = subimages.transpose(0, 1, 3, 2, 4)
    subimages = subimages.reshape(-1, side_size, side_size)
    
    return subimages

class ConfigurableAutoencoder(nn.Module): # La clase Autoencoder hereda de la clase nn.Module, que es una clase base para todos los modelos en PyTorch.
                                          # Esto permite que nuestra clase Autoencoder tenga todas las funcionalidades necesarias para ser un modelo de aprendizaje profundo en PyTorch.
    
    def __init__(self, config: dict, image_size: int = None):
        super(ConfigurableAutoencoder, self).__init__()
        
        self.config = config
        
        if image_size is not None:
            if not isinstance(image_size, int) or image_size <= 0:
                raise ValueError("image_size debe ser un entero positivo")
            self.image_size = image_size
        else:
            if 'training' not in config:
                raise KeyError("Config debe tener sección 'training' o se debe proveer image_size")
            required_params = ['n_cuad_lado', 'pixeles_cuad']
            missing = [p for p in required_params if p not in config['training']]
            if missing:
                raise KeyError(
                    f"Config debe tener los parámetros {missing} en 'training' "
                    f"o se debe proveer image_size directamente"
                )
            self.image_size = self.config['training']['n_cuad_lado'][0] * self.config['training']['pixeles_cuad'][0]
        
        self.flat_size = self.image_size * self.image_size
        self.encoding_dim = self.config['model']['encoding_dim']
        
        self.encoder = self._build('encoder')
        self.decoder = self._build('decoder')
    
    def conv2d_output_size(self, input_size, kernel_size, stride, padding):
        return ((input_size + 2 * padding - kernel_size) // stride) + 1
    
    def conv2d_transpose_output_size(self, input_size, kernel_size, stride, padding):
        return (input_size - 1) * stride - 2 * padding + kernel_size
        
    def _build(self, component: str) -> nn.Sequential:
        layers = []
        last_out_channels = 1
        current_size = self.image_size
        
        if component == 'encoder':
            input_dim = self.flat_size
            component_layers = self.config['encoder']['layers']
        elif component == 'decoder':
            input_dim = self.encoding_dim
            component_layers = self.config['decoder']['layers']
        else:
            raise ValueError(f"El parámetro component solo puede ser 'encoder' o 'decoder', se recibió: {component}")
        
        for layer in component_layers:
            
            if layer.type == "flatten":
                input_dim = current_size * current_size * last_out_channels
                layers.append(nn.Flatten())
                last_out_channels = 1
                
            elif layer.type == "dense":
                layers.append(nn.Linear(
                    input_dim,
                    layer['dim'],
                    bias=layer.get('bias', True)))
                input_dim = layer['dim']
                
            elif layer.type == "unflatten":
                layers.append(nn.Unflatten(1, (layer.get('out_channels', 1), layer['dim1'], layer['dim2'])))
                current_size = layer['dim1']  # Asumiendo imágenes cuadradas
                last_out_channels = layer.get('out_channels', 1)
                
            elif layer.type == 'conv2d':
                layers.append(nn.Conv2d(
                    in_channels=layer.get('in_channels', last_out_channels),
                    out_channels=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    stride=layer.get('stride', 1),
                    padding=layer.get('padding', 0)
                ))
                stride = layer.get('stride', 1)
                padding = layer.get('padding', 0)
                kernel_size = layer['kernel_size']
                current_size = self.conv2d_output_size(current_size, kernel_size, stride, padding)
                last_out_channels = layer['filters']
            
            elif layer.type == 'conv2d_transpose':
                layers.append(nn.ConvTranspose2d(
                    in_channels=layer.get('in_channels', last_out_channels),
                    out_channels=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    stride=layer.get('stride', 1),
                    padding=layer.get('padding', 0)
                ))
                stride = layer.get('stride', 1)
                padding = layer.get('padding', 0)
                kernel_size = layer['kernel_size']
                current_size = self.conv2d_transpose_output_size(current_size, kernel_size, stride, padding)
                last_out_channels = layer['filters']
                
            elif layer.type == 'maxpool2d':
                pool_size = layer['pool_size']
                layers.append(nn.MaxPool2d(
                    kernel_size=pool_size,
                    stride=layer.get('stride', None),
                    padding=layer.get('padding', 0)
                ))
                current_size = current_size // pool_size

            elif layer.type == 'upsample':
                scale_factor = layer['scale_factor']
                layers.append(nn.Upsample(
                    scale_factor=scale_factor,
                    mode=layer.get('mode', 'nearest')
                ))
                current_size = current_size * scale_factor

            activation = layer.get('activation', '').lower()
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded