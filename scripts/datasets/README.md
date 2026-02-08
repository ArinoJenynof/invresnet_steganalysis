# Steganalysis Dataset Generation Scripts
What I used, and did, to generate steganographic images. Images are sourced from BOWS2 which can be found [here](https://web.archive.org/web/20130510185330/http://bows2.ec-lille.fr/BOWS2OrigEp3.tgz), and BossBase which can be found [here](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip).

### Processing Steps
1. Resize all images to 256×256 (from 512×512), using Nearest-Neighbour. I used XnConvert here. Output as PNG to save some space.
2. Use Octave code to create steganographic images, this step outputs to .mat files.
3. Load .mat files and convert to PNG again.

# Steganographic Algorithms
Most of the code came from DDE Dinghamton's [download section](https://dde.binghamton.edu/download/). If not from there then it's from daniellerch's [aletheia-external-resource](https://github.com/daniellerch/aletheia-external-resources). I just lightly modified it so it runs in Octave, only one line change really.

### Running the scripts
Make sure you have Octave with `parallel` package installed, and Python with `scipy` to load the .mat files + `PIL` to save the images to PNG. Then just open Octave, `cd` to scripts directory, and type like this:
```octave
pkg load image; pkg load signal; pkg load nan; pkg load parallel;
images_cover = dir('./images_cover/*.png');
for i = 1:rows(images_cover)
  images_cover(i).payload = single(<YOUR_PAYLOAD_SETTINGS>);
endfor
pararrayfun(nproc - 1, @parallel_<STEGO_ALGO>, images_cover);
```
#### ATTENTION
1. <YOUR_PAYLOAD_SETTING> is in bits per pixel (bpp), don't forget to set it.
2. <STEGO_ALGO> currently is one of S_UNIWARD, WOW, or HUGO, so set accordingly.
3. I still don't know how to make a script that just launch with Octave, not familiar with the language, sorry.

## About the Octave Scripts
The script use [parallel package](https://gnu-octave.github.io/packages/parallel/) to speed up the embedding process. This way was chosen, because I had some difficulties running aletheia. Digging further, it seems like [subprocess.Popen(..., shell=True)](https://github.com/daniellerch/aletheia/blob/71805e7419936dedb8df07b98ff117a30ee654ed/aletheialib/octave_interface.py#L119) doesn't play nicely on Windows. It returns immediately, not waiting for the .mat files to finished processing. So I'm looking for the alternatives.

Next there is a Python package called [Oct2Py](https://github.com/blink1073/oct2py). It does work, but slow, about 6 hours for 10.000 images, per one payload setting, and I need to generate a lot of them. Oct2Py use files to communicate between Python and Octave, so I guess that's where the bottleneck is. Figures that I need to do multiprocessing in Octave itself, and thankfully `parallel` package delivers. It cuts down to 10-30 minutes(!!) per one payload settings. Except HUGO, more on that later.

I mimicked the structure from DDE itself, so put your cover images into `images_cover` folder and it will be outputted to `images_stego` folder. Run the Octave script first, then the Python script. No command line interface given, so just change the settings straight in the source.

### Notes
HUGO is slow, cursory look indicates that it uses loops instead of vectorised operations. It also has higher iteration limits (100) compared to S_UNIWARD or WOW (30).