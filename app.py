import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import numpy as np
import sys

from python_files.Noise_Reduction_Datagen_paths import Signal_Synthesis_DataGen
from python_files.unet_basic import Model


