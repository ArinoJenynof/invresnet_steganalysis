function embed(x = argv())
	method = x{1};
	payload = str2double(x{2});
	
	pkg load parallel;
	images_cover = dir('./images_cover/*');
	for i = 1:rows(images_cover)
		images_cover(i).method = method;
		images_cover(i).payload = single(payload);
	end
	
	pararrayfun(nproc ./ 2 + 1, @embed_worker, images_cover);
end
