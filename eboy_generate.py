from absl import app
from absl import flags
from absl import logging
from hashlib import md5
from io import BytesIO
from json import load as json_load
from math import ceil
from os import makedirs
from os.path import isdir
from PIL import Image
from re import compile as re_compile
from requests import get

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 128, 'The size of the square images to crop.')
flags.DEFINE_integer('stride', 32, 'The stride of the sliding crop window.')
flags.DEFINE_boolean('extend_partial', True,
                     'Whether to extend partial crops to full squares.')
flags.DEFINE_float('max_empty', 0.5,
                   'The maximum relative size of the largest empty area.')
flags.DEFINE_string('input_data', 'eboy_data.json',
                    'The file containing the source image URLs.')
flags.DEFINE_string('images_dir', 'eboy-images',
                    'The image directory in which to save the crops.')
flags.DEFINE_string('image_format', 'png',
                    'The format under which to save images.')

SCALE_PATTERN = re_compile(r'^.*-(\d+)x.png$')


def main(_):
    if not isdir(FLAGS.images_dir):
        makedirs(FLAGS.images_dir)

    with open(FLAGS.input_data) as json_file:
        for image_url in json_load(json_file)['image_urls']:
            logging.info(f'Processing {image_url}')
            image = Image.open(BytesIO(get(image_url).content)).convert('RGB')
            image_hash = md5(image_url.encode()).hexdigest()

            # Skip images smaller than the crop size.
            if image.width < FLAGS.size or image.height < FLAGS.size:
                logging.warning(f'Skipping {image.width}x{image.height} image')
                continue

            # Resize to pixel size of 1, if needed.
            scale_match = SCALE_PATTERN.match(image_url)
            if scale_match:
                scale = int(scale_match.group(1))
                logging.warning(f'Resizing by {scale}x')
                image = image.resize((image.width // scale,
                                      image.height // scale), Image.NEAREST)

            # Divide the image into squares. If it doesn't evenly divide,
            # optionally shift the last crops in to avoid empty areas.
            for y in range(ceil(image.height / FLAGS.stride)):
                y_min = y * FLAGS.stride
                y_max = y * FLAGS.stride + FLAGS.size

                if y_max > image.height:
                    if not FLAGS.extend_partial:
                        continue
                    y_max = image.height
                    y_min = y_max - FLAGS.size

                for x in range(ceil(image.width / FLAGS.stride)):
                    x_min = x * FLAGS.stride
                    x_max = x * FLAGS.stride + FLAGS.size

                    if x_max > image.width:
                        if not FLAGS.extend_partial:
                            continue
                        x_max = image.width
                        x_min = x_max - FLAGS.size

                    # Create the cropped image.
                    crop = image.crop((x_min, y_min, x_max, y_max))

                    # Discard predominantly empty crops by calculating the
                    # fraction of the crop filled with the most frequent color.
                    num_pixels = crop.width * crop.height
                    colors = crop.getcolors(num_pixels)
                    primary_count = max([count for count, _ in colors])
                    if primary_count > FLAGS.max_empty * num_pixels:
                        logging.warning('Skipping empty crop')
                        continue

                    # Save the image
                    name = f'{FLAGS.images_dir}/{image_hash}_{y}_{x}'
                    name += f'.{FLAGS.image_format}'
                    crop.save(name, FLAGS.image_format)
                    logging.info(f'Saved {name}')


if __name__ == '__main__':
    app.run(main)
