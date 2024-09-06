
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.utils import load_restore_ckpt, load_embedder_ckpt
import os
from gradio_imageslider import ImageSlider

# Enforce CPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedder_model_path = "ckpts/embedder_model.tar"  # Update with actual path to embedder checkpoint
restorer_model_path = "ckpts/onerestore_cdd-11.tar"   # Update with actual path to restorer checkpoint

# Load models on CPU only
embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=embedder_model_path)
restorer = load_restore_ckpt(device, freeze_model=True, ckpt_name=restorer_model_path)

# Define image preprocessing and postprocessing
transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 


def postprocess_image(tensor):
    image = tensor.squeeze(0).cpu().detach().numpy()
    image = (image) * 255  # Assuming output in [-1, 1], rescale to [0, 255]
    image = np.clip(image, 0, 255).astype("uint8")  # Clip values to [0, 255]
    return Image.fromarray(image.transpose(1, 2, 0))  # Reorder to (H, W, C)

# Define the enhancement function
def enhance_image(image, degradation_type=None):
    # Preprocess the image
    input_tensor = torch.Tensor((np.array(image)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    lq_em = transform_resize(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    lq_em = transform_resize(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate embedding
    if degradation_type == "auto" or degradation_type is None:
        text_embedding, _, [text] = embedder(lq_em, 'image_encoder')
    else:
        text_embedding, _, [text] = embedder([degradation_type], 'text_encoder')
    
    # Model inference
    with torch.no_grad():
        enhanced_tensor = restorer(input_tensor, text_embedding)
    
    # Postprocess the output
    return (image, postprocess_image(enhanced_tensor)), text

# Define the Gradio interface
def inference(image, degradation_type=None):
    return enhance_image(image, degradation_type)

#### Image,Prompts examples
examples = [
            ['image/low_haze_rain_00469_01_lq.png'],
            ['image/low_haze_snow_00337_01_lq.png'],
            ]



# Create the Gradio app interface using updated API
interface = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="pil", value="image/low_haze_rain_00469_01_lq.png"),  # Image input
        gr.Dropdown(['auto', 'low', 'haze', 'rain', 'snow',\
                                            'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                                    'haze_snow', 'low_haze_rain', 'low_haze_snow'], label="Degradation Type", value="auto")  # Manual or auto degradation
    ],
    outputs=[
        ImageSlider(label="Restored Image", 
                        type="pil",
                        show_download_button=True,
                        ),  # Enhanced image outputImageSlider(type="pil", show_download_button=True, ),
        gr.Textbox(label="Degradation Type")  # Display the estimated degradation type
    ],
    title="Image Restoration with OneRestore",
    description="Upload an image and enhance it using OneRestore model. You can choose to let the model automatically estimate the degradation type or set it manually.",
    examples=examples,
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
