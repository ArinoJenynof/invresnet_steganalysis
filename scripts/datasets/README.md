# Steganalysis Dataset Generation Scripts
What I used, and did, to generate steganographic images. Images are sourced from BOWS2 which can be found [here](https://web.archive.org/web/20130510185330/http://bows2.ec-lille.fr/BOWS2OrigEp3.tgz), and BossBase which can be found [here](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip).

### Processing Steps
1. Resize all images to 256×256 (from 512×512), using whatever downscaling method you prefer. I used XnConvert here. Output as PNG to save some space.
2. Use `embed.m` Octave script to create steganographic images.

# Steganographic Algorithms
Most of the code came from DDE Binghamton's [download section](https://dde.binghamton.edu/download/). If not from there then it's from daniellerch's [aletheia-external-resource](https://github.com/daniellerch/aletheia-external-resources). I just lightly modified it so it runs in Octave, only one line change really.

### Running the scripts
As of Octave 10, `parallel` package, which I used before, is not buildable due to bumped C++ standard to C++17. I tried to compile `parallel` with `CXXFLAGS='-std=gnu++17'` and other flags but that just creates more errors down the line. So I tried to find other way to parallelise embedding script.

Enter GNU Parallel. Along with better `embed.m` script so now it can be run in command line like a normal `.cmd`/`.sh` script or called inside Octave interpreter. Though if you call it inside Octave interpreter then you need to find a way to parallelise it *inside* the interpreter yourself.

#### Prerequisites
The official Windows [download](https://octave.org/download#ms-windows) do use some kind of MSYS2. I don't know if it keeps `pacman`, which you need to install GNU Parallel. If it doesn't, just use [MSYS2](https://github.com/msys2/msys2-installer/releases/latest) directly. If it does, the steps described here should work just as well. To setup Octave inside MSYS2 generally you need to do:
1. Open `UCRT64` environment.
2. Update packages and install build dependencies.
3. Install `mingw-w64-ucrt-x86_64-octave` and `parallel` (you can use other environment like CLANG64 if you prefer, should be the same all around. Probably choking on GNU extension because, well, this is GNU Octave)
4. Enter octave and install `image` package.
5. Either still inside interpreter, run `embed.m`. Or exit and use GNU Parallel to run `embed.m`

The commands will probably look like this:
```bash
pacman -Syuu --noconfirm
pacman -S --needed make parallel mingw-w64-ucrt-x86_64-{gcc,octave}
octave-cli
```
You will then enter octave interpreter:
```octave
pkg install image;
exit;
```
Now run with GNU Parallel to speed things up:
```bash
parallel -j 2 -k octave embed.m <steganograhy method> {} <payload> ::: <list of images cover>
```

#### Some explanations
1. `<steganography method>` is currently `S_UNIWARD` or `WOW`. More may be added in the future.
2. `<payload>` is floating point number, in bits per pixel. Bigger payload == bigger distortion.
3. `<list of images cover>` is just images path you want to embed. Put `images_cover/*` to mimic the structure of the original code from DDE.

## Historical Notes
I had some difficulties running Aletheia due to [subprocess.Popen(..., shell=True)](https://github.com/daniellerch/aletheia/blob/71805e7419936dedb8df07b98ff117a30ee654ed/aletheialib/octave_interface.py#L119) doesn't play nicely on Windows. It returns immediately, not waiting for the .mat files to finished processing.

Next there is a Python package called [Oct2Py](https://github.com/blink1073/oct2py). It does work, but slow, about 6 hours for 10.000 images, per one payload setting, and I need to generate a lot of them. Oct2Py use files to communicate between Python and Octave, so I guess that's where the bottleneck is.

I mimicked the structure from DDE itself, so put your cover images into `images_cover` folder and it will be outputted to `images_stego` folder.

HUGO is slow, cursory look indicates that it uses loops instead of vectorised operations. It also has higher iteration limits (100) compared to S_UNIWARD or WOW (30).