import argparse
import nibabel
import numpy as np
from skimage import morphology


def segment_bones(input, output, threshold, closing_radius):
    # Read the image from the file
    nib_img = nibabel.load(input)

    # Run the binary thresholding operation on the raw numpy image
    binary_img = nib_img.get_data() > threshold

    # Run binary closing using a ball-shaped structuring element
    closed_img = morphology.binary_closing(binary_img, selem=morphology.ball(radius=closing_radius))

    # Right now our pixel type is "bool" -- Nifti doesn't support that, so we cast it
    # to an image of 1-byte pixels containing either 1 or 0
    eight_bit = closed_img.astype(np.uint8)

    # Write the image to the output path using the same affine transform as the original image
    nibabel.Nifti1Image(eight_bit, nib_img.affine).to_filename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='bone_seg', description='Segment bones from non-contrast CT images.')

    parser.add_argument('--input', help='The path to the input image.', required=True)
    parser.add_argument('--output', help='Where to write the output image.', required=True)
    parser.add_argument('--threshold', help='Threshold value in HU', type=int,
                        required=False, default=200)
    parser.add_argument('--closing-radius',
                        help='Radius of the structuring element for binary closing.',
                        type=int, required=False, default=2)

    args = parser.parse_args()

    segment_bones(args.input, args.output, args.threshold, args.closing_radius)
