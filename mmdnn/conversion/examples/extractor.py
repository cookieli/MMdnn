#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

from __future__ import absolute_import
from six import text_type as _text_type
import argparse

def get_command_line_args(network_choices):
    '''read arguments from command line
        network_choices means a list of string which is network names'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network',
                        type=_text_type, help='Model Type', required=True,
                        choices=network_choices)

    parser.add_argument('-i', '--image',
                        type=_text_type, help='Test Image Path')

    return parser.parse_args()

class base_extractor(object):

    def __init__(self):
        pass


    @classmethod
    def help(cls):
        print('Supported models: {}'.format(list(cls.architecture_map.keys())))


    @classmethod
    def sanity_check(cls, architecture):
        if architecture is None:
            cls.help()
            return False

        elif not architecture in cls.architecture_map:
            cls.help()
            raise ValueError("Unknown pretrained model name [{}].".format(architecture))

        else:
            return True

    @classmethod
    def download(cls, architecture):
        raise NotImplementedError()


    @classmethod
    def inference(cls, image_path):
        raise NotImplementedError()
