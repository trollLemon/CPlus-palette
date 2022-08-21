# CPlus-pallete
Command line utility that generates a color pallet based on an image inputed by the user. The amount of colors can also be specified.

# Usage

The program has two inputs, a path to an image file (it can be a png or a jpg), and how many colors you want in your
pallete.

For example:

```bash
$ ./cpluspallet ~/Pictures/apicture.png 5
```
will generate 5 colors based off the colors in apicture.png.

The output will look like the following:

![image](https://user-images.githubusercontent.com/90001607/185774728-5e7a760c-e054-4ff5-837a-a7470e927a80.png)

These are colors generated from this image:

![image](https://user-images.githubusercontent.com/90001607/185774623-82d15335-8d70-444a-83ff-15c2b0006ec6.png)

These are the colors outputed from the program:

![#4aee30](https://user-images.githubusercontent.com/90001607/185774744-4b3ebd2d-4411-400c-ab88-cef6e5c59231.png)
![#7b80a1](https://user-images.githubusercontent.com/90001607/185774745-f2c34367-3c6d-46d7-a66b-e873e90791fe.png)
![#28ccff](https://user-images.githubusercontent.com/90001607/185774747-6cfe540f-4cad-4dfe-ae98-b1f7d618ee44.png)
![#44e21e](https://user-images.githubusercontent.com/90001607/185774748-837e0a14-7b68-484d-a085-c98438a6880c.png)
![#cab0db](https://user-images.githubusercontent.com/90001607/185774749-fef2b515-6b9b-4e64-b2ab-3607db21704d.png)


It also works with other image files as well:
This was saved as a jpg image.
![7255636](https://user-images.githubusercontent.com/90001607/185774868-909d7f8c-19cc-4272-a304-63691b3d5d2b.jpg)
##### (I got this image off wallpaperaccess: https://wallpaperaccess.com/3440x1440-minimal, scroll down a bit and you'll find it.)

and here are 8 colors generated:

![#1d1f24](https://user-images.githubusercontent.com/90001607/185775052-7cf9e981-3ca3-4250-8649-7b2fe86175fb.png)
![#3ec8d3](https://user-images.githubusercontent.com/90001607/185775053-834c2a85-29c3-4d97-b338-8192cb05e1d3.png)
![#75e9f3](https://user-images.githubusercontent.com/90001607/185775054-e2e91f5f-74bc-464a-9edf-aa1c72b660d0.png)
![#262a33](https://user-images.githubusercontent.com/90001607/185775055-e57bf43d-9452-42ee-a987-95b6e4530838.png)
![#75596f](https://user-images.githubusercontent.com/90001607/185775056-2738735b-c59e-42ca-a4b7-576b01346c10.png)
![#141518](https://user-images.githubusercontent.com/90001607/185775057-9fd3e4eb-6a9d-49fa-9434-d6db04441f45.png)
![#343949](https://user-images.githubusercontent.com/90001607/185775058-bedd0e4f-f7a9-4426-9e15-4667de8cb7c8.png)
![#f19aa9](https://user-images.githubusercontent.com/90001607/185775059-a2abfbf4-1db0-4993-99ae-e17502eef844.png)


You can have the program generate as many colors you want

# Building
You will need the CImg in order to build this project.

You can follow the install instructions from CImg's main page: 
https://cimg.eu/download.html

Or if you are on linux, you can install it using the package manager of your choice.

On arch, you can install CImg from the AUR with the following:
```bash
$ yay CImg
```
and choose the option *community/cimg*.

On Debain (or mint, ubuntu, other debian based distros), you will need to get imagemagick:
```bash
$  sudo apt install imagemagick
```
and CImg:
```bash
$  sudo apt install cimg-dev
```
Once you have gotten CImg, you can now compile and link the project.

```bash 
$ g++ -o cpluspallet src/main.cpp src/colors.cpp -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
```
 Running this will create a binary called 'cpluspallet'.

# References

These stack overflow articles:
https://stackoverflow.com/questions/14375156/how-to-convert-a-rgb-color-value-to-an-hexadecimal-value-in-c
https://stackoverflow.com/questions/5823854/how-can-i-generate-a-palette-of-prominent-colors-from-an-image

And these:
https://sighack.com/post/averaging-rgb-colors-the-right-way
https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-cpp/
