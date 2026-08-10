#! /bin/octave -qf
function embed (cover_path = argv (){1}, method = argv (){2}, payload = str2double (argv (){3}))
	method = validatestring(method, {"S_UNIWARD", "WOW"});
	mustBePositive (payload);

	pkg load image;
	addpath ("./steganography");
	stego = feval (method, cover_path, payload);
	[in_dir, in_name, in_ext] = fileparts (cover_path);
	outpath = fullfile ("./stegos", [in_name in_ext]);
	imwrite (uint8 (stego), outpath);
	disp (outpath);
end
