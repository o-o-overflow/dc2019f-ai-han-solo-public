#!/usr/bin/env python3

#composite -gravity west -geometry +28+0 emnist-png/emnist-balanced/1/image_10006.png 0.png o.png

import ai_han_solo
import contextlib
import itertools
#import tempfile
import random
import pygpar
import shutil
import os

ALPHABET="0123456789ABCDEF"

def overload_cmd(out_path, src_path, paste_path, offset):
	return f"composite -gravity northwest -geometry +{offset*28}+0 {src_path} {paste_path} {out_path}"

def many_pastes(out_dir, src_path, paste_paths, offset):
	return [ overload_cmd(os.path.join(out_dir, f"{i}.png"), src_path, paste_path, offset) for i,paste_path in enumerate(paste_paths) ]

def paste_letter(out_dir, src_path, fonts_path, letter, variants, offset):
	paste_paths = random.sample(os.listdir(os.path.join(fonts_path, letter)), variants)
	return many_pastes(out_dir, src_path, paste_paths, offset)

def make_candidates(width, variants, candidate_path="/dev/shm/candidates", fonts_path="emnist-png/emnist-balanced"):
	with contextlib.suppress(FileExistsError):
		os.makedirs(candidate_path)
	possibilities = list(itertools.product(ALPHABET, repeat=width))
	letterforms = { letter: os.listdir(os.path.join(fonts_path, letter)) for letter in ALPHABET }

	arglists = [ ]
	for p in possibilities:
		for n in range(variants):
			arglists.append([ os.path.join(candidate_path, f"candidate-{''.join(p)}-{n}.png") ] + [ os.path.join(fonts_path, c, random.choice(letterforms[c])) for c in p ])

	args = " ".join(f"{{{d+2}}}" for d in range(width))
	with pygpar.PP(f"montage {args} -tile {width}x1 -geometry +0+0 {{{1}}}", filter_exists=True, eta=True, jobs=40) as _p:
		_p.queue_list(arglists)

	return next(iter(zip(*arglists)))

def make_guesses(src_paths, width, variants, offset, candidates_path="/dev/shm/candidates", guess_path="/dev/shm/guesses", fresh=True, **kwargs):
	candidates = make_candidates(width, variants, candidate_path=candidates_path, **kwargs)
	with contextlib.suppress(FileExistsError):
		os.makedirs(guess_path)

	arglists = [ ]
	for sp in src_paths:
		for c in candidates:
			filename = "guess_%s_%s.png" % (sp.split("out_")[-1].split(".png")[0], c.split('candidate-')[-1].split('.')[0])
			arglists.append([ os.path.join(guess_path, filename), c, sp ])
	with pygpar.PP(f"composite -gravity northwest -geometry +{offset*ai_han_solo.LETTER_WIDTH}+0 {{{2}}} {{{3}}} {{{1}}}", filter_exists=not fresh, eta=True, jobs=40) as _p:
		_p.queue_list(arglists)

	return next(iter(zip(*arglists)))

def evaluate_guesses(target_class, src_path, guess_path="/dev/shm/guesses", regenerate=True, **kwargs):
	if regenerate:
		with contextlib.suppress(FileNotFoundError):
			shutil.rmtree(guess_path)

	orig_predictions = ai_han_solo.predict(*ai_han_solo.expand_paths([src_path]))
	orig_values = { }
	for path,values in orig_predictions.items():
		out_id = path.split("out_")[-1].split(".png")[0]
		values = list(map(float, values))
		orig_values[out_id] = values

	make_guesses(ai_han_solo.expand_paths([src_path]), guess_path=guess_path, fresh=regenerate, **kwargs)
	predictions = ai_han_solo.predict(*ai_han_solo.expand_paths([guess_path]))
	guess_vals = { }
	target_vals = [ ]
	for path,values in predictions.items():
		guess = path.split("_")[-1].split("-")[0]
		orig_id = path.split("guess_")[-1].rsplit("_", 1)[0]
		orig_val = orig_values[orig_id][target_class]
		new_val = values[target_class]

		delta = new_val - orig_val
		guess_vals.setdefault(guess, []).append(delta)
		target_vals.append([delta, guess])

	avgs = { g:sum(v) for g,v in guess_vals.items() }
	print("# Results:")
	print("... avg:", sorted(avgs.items(), key=lambda i: -i[1])[:10])
	print("... max:", sorted(target_vals, key=lambda i: -i[0])[:10])

if __name__ == '__main__':
	for i in range(16):
		imgattack.evaluate_guesses(16, "output", width=1, variants=128, offset=i)
		print(f"# OFFSET {i} ^^^^")
