import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image 

def generate_test_cases(image_path, prompt_template, num_cases=5):
    """
    Generates test cases by providing a prompt with different variations based on the input image using CLIP. 

    Args:
        image_path: Path to the image. 
        prompt_template: A string template for the test case prompt, including placeholders for variations. 
        num_cases: Number of test cases to generate. 
    
    Returns:
        A list of generated test case prompts. 
    """
    
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load image 
    image = Image.open(image_path)
    
    # Encode image 
    image_input = processor(images=image, return_tensors="pt")
    image_embeddings = model.get_image_features(**image_input)
    
    test_cases = []
    for i in range(num_cases):
        # Generate variations for the prompt
        variation = " "
        # Add logic to generate different variations for your prompt template
        
        # Combine prompt with variation
        test_prompt = prompt_template.format(variation) 
        
        #test_prompt = prompt_template.format()

        # Calculate similarity with image embedding
        text_input = processor(text=[test_prompt], return_tensors="pt")
        text_embeddings = model.get_text_features(**text_input)
        similarity = torch.cosine_similarity(image_embeddings, text_embeddings, dim=1).item() 
        
        # Store test case with similarity score 
        test_cases.append((test_prompt, similarity))
    
    return test_cases
 
# Example usage
image_path = "zero.jpg" 
prompt_template = "Test case: User clicks on the {} button"
generated_cases = generate_test_cases(image_path, prompt_template, num_cases=10) 

for case in generated_cases:
    print(case[0], " - Similarity Score:", case[1])