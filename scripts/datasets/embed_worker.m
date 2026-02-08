function embed_worker(image)
	pkg load image;
	f = str2func(image.method);
	stego = f(fullfile(image.folder, image.name), image.payload);
	imwrite(uint8(stego), fullfile("./images_stego", image.name));
	disp(image.name);
end
