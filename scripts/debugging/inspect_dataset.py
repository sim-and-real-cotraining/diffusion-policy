import zarr
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    # Path to your Zarr directory
    zarr_directory = 'data/planar_pushing_cotrain/real_world_tee_data.zarr'

    # Open the Zarr root directory
    zarr_root = zarr.open(zarr_directory, mode='r')

    # Access the overhead_camera group
    data_group = zarr_root['data']
    meta_group = zarr_root['meta']

    # Access the overhead_camera dataset within the data group
    overhead_camera_dataset = data_group['overhead_camera']
    wrist_camera_dataset = data_group['wrist_camera']
    episode_ends = meta_group['episode_ends']

    # Retrieve the overhead images
    real_overhead_image = overhead_camera_dataset[episode_ends[10]-1]
    sim_overhead_image = cv2.imread('/home/adam/workspace/planning-through-contact/trajectories_rendered/sim_tee_data/1/overhead_camera/31500.png')
    sim_overhead_image = cv2.cvtColor(sim_overhead_image, cv2.COLOR_BGR2RGB)

    # Retrieve the wrist images
    real_wrist_image = wrist_camera_dataset[episode_ends[10]-1]
    sim_wrist_image = cv2.imread('/home/adam/workspace/planning-through-contact/trajectories_rendered/sim_tee_data/1/wrist_camera/31500.png')
    sim_wrist_image = cv2.cvtColor(sim_wrist_image, cv2.COLOR_BGR2RGB)

    # Process all images
    # kernel_size = (9, 9)
    # pixel_size = 4
    # levels = 32
    # real_overhead_image = process_image(real_overhead_image, kernel_size, pixel_size, levels)
    # sim_overhead_image = process_image(sim_overhead_image, kernel_size, pixel_size, levels)
    # real_wrist_image = process_image(real_wrist_image, kernel_size, pixel_size, levels)
    # sim_wrist_image = process_image(sim_wrist_image, kernel_size, pixel_size, levels)

    # Define the transparency (alpha) values for both images (0.0 to 1.0)
    alpha_real_image = 0.5
    alpha_sim_image = 1-alpha_real_image

    # Overlay the images
    overlay_overhead_image = cv2.addWeighted(real_overhead_image, alpha_real_image, sim_overhead_image, alpha_sim_image, 0)
    overlay_wrist_image = cv2.addWeighted(real_wrist_image, alpha_real_image, sim_wrist_image, alpha_sim_image, 0)
    
    # Display the images using subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    # Overhead camera
    axs[0][0].imshow(real_overhead_image)
    axs[0][0].set_title('Real Overhead Camera')
    axs[0][0].axis('off')  # Hide axes

    axs[0][1].imshow(sim_overhead_image)
    axs[0][1].set_title('Sim Overhead Camera')
    axs[0][1].axis('off')  # Hide axes

    axs[0][2].imshow(overlay_overhead_image)
    axs[0][2].set_title('Overlayed Overhead Camera')
    axs[0][2].axis('off')  # Hide axes

    # Wrist camera
    axs[1][0].imshow(real_wrist_image)
    axs[1][0].set_title('Real Wrist Camera')
    axs[1][0].axis('off')  # Hide axes

    axs[1][1].imshow(sim_wrist_image)
    axs[1][1].set_title('Sim Wrist Camera')
    axs[1][1].axis('off')  # Hide axes

    axs[1][2].imshow(overlay_wrist_image)
    axs[1][2].set_title('Overlayed Wrist Camera')
    axs[1][2].axis('off')  # Hide axes


    # Save the overlayed image
    plt.savefig('overlayed_image.png', bbox_inches='tight', pad_inches=0)

    plt.show()

def blur_image(image, kernel_size=(5, 5)):
    return cv2.blur(image, kernel_size, 0)

def pixelate(image, pixel_size):
    # Resize to a smaller size
    small_image = cv2.resize(image, (image.shape[1] // pixel_size, image.shape[0] // pixel_size), interpolation=cv2.INTER_LINEAR)
    # Resize back to the original size
    pixelated_image = cv2.resize(small_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

def quantize(image, levels):
    quantized_image = (image // (256 // levels)) * (256 // levels)
    return quantized_image

def process_image(image, kernel_size, pixel_size, levels):
    # Blur the image
    blurred_image = blur_image(image, kernel_size)
    # Pixelate the image
    pixelated_image = pixelate(blurred_image, pixel_size)
    # Quantize the image
    quantized_image = quantize(pixelated_image, levels)
    return quantized_image

if __name__ == "__main__":
    main()