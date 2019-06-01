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
from requests import get

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 128, 'The size of the square images to crop.')
flags.DEFINE_integer('stride', 64, 'The stride of the sliding crop window.')
flags.DEFINE_string('input_data', 'eboy_data.json',
                    'The file containing the source image URLs.')
flags.DEFINE_string('images_dir', 'eboy-images',
                    'The image directory in which to save the crops.')
flags.DEFINE_string('image_format', 'png',
                    'The format under which to save images.')


def main(_):
    if not isdir(FLAGS.images_dir):
        makedirs(FLAGS.images_dir)

    with open(FLAGS.input_data) as json_file:
        for image_url in json_load(json_file)['image_urls']:
            image = Image.open(BytesIO(get(image_url).content))
            image_hash = md5(image_url.encode()).hexdigest()

            # TODO: Figure out pixel scale and resize.

            # Divide the image into squares. If it doesn't evenly divide, shift
            # the last crops in to avoid empty areas.
            for y in range(ceil(image.height / FLAGS.stride)):
                y_min = y * FLAGS.stride
                y_max = y * FLAGS.stride + FLAGS.size

                if y_max > image.height:
                    y_max = image.height
                    y_min = y_max - FLAGS.size

                for x in range(ceil(image.width / FLAGS.stride)):
                    x_min = x * FLAGS.stride
                    x_max = x * FLAGS.stride + FLAGS.size

                    if x_max > image.width:
                        x_max = image.width
                        x_min = x_max - FLAGS.size

                        crop = image.crop((x_min, y_min, x_max, y_max))
                        name = '%s/%s_%d_%d.%s' % (FLAGS.images_dir,
                                                   image_hash, y, x,
                                                   FLAGS.image_format)
                        crop.save(name, FLAGS.image_format)
                        logging.info('Saved %s' % name)


if __name__ == '__main__':
    app.run(main)
