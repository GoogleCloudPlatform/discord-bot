from vertexai.generative_models import FunctionDeclaration, Part
from vertexai.preview.vision_models import ImageGenerationModel, GeneratedImage

generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")


def generate_image(prompt: str) -> GeneratedImage:
    """
    Use Imagen 3 to generate a single image according to provided prompt. The image will be attached to the next message that will be sent.

    Args:
        prompt: Textual description of the image that is to be generated.

    Returns:
        The image generated according to prompt. It will be automatically attached to the next message.
    """
    image = generation_model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="1:1",
    )
    img = image[0]
    return img


generate_image_tool = FunctionDeclaration.from_func(generate_image)


def call_generate_image(part: Part) -> (str, GeneratedImage):
    assert part.function_call.name == "generate_image"
    return (
        "Your image was generated successfully and will be attached to your next message.",
        generate_image(part.function_call.args["prompt"]),
    )
