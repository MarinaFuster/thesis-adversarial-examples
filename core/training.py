# set the matplotlib backend so figures can be saved in the background
import os

import matplotlib

from modules.autoencoder import Autoencoder
from modules.data_loader import load_data
from modules.model_loader import ModelLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2


def train_autoencoder(
		epochs,
		batch_size,
		autoencoder,
		loss_plot_name,
		decoder_plot_name,
		output_folder,
		save=True
):
	"""
	In addition to a loss plot, there will be a loss csv file for test loss
	for each epoch.
	"""
	loss_plot = f'{output_folder}/{loss_plot_name}.png'
	decoder_plot_test = f'{output_folder}/{decoder_plot_name}_test_samples.png'
	decoder_plot_train = f'{output_folder}/{decoder_plot_name}_train_samples.png'

	print("[INFO] loading personal dataset...")
	train_data, test_data, _, _ = load_data(mode='gray')

	# shuffles both train and test datasets, done in-place
	np.random.shuffle(train_data)
	np.random.shuffle(test_data)

	# re-scale the pixel intensities to the range [0, 1]
	train_data = train_data.astype("float32") / 255.0
	test_data = test_data.astype("float32") / 255.0

	print("[INFO] building autoencoder...")
	autoencoder.train(epochs, batch_size, train_data, test_data)

	H = autoencoder.H

	N = np.arange(0, epochs)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(N, H.history["loss"], label="train_loss")
	plt.plot(N, H.history["val_loss"], label="val_loss")
	plt.title("Training and Test Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(loss_plot)

	avg = None
	if epochs >= 10:
		print("Avg. test dataset loss for last 10 epochs")
		avg = sum(H.history["val_loss"][-10:])/10
		print(avg)

	# saves loss for each epoch for test data
	np.savetxt(f"{output_folder}/{loss_plot_name}_test_data.csv", H.history['val_loss'], delimiter=",")
	# saves loss for each epoch for train data
	np.savetxt(f"{output_folder}/{loss_plot_name}_train_data.csv", H.history['loss'], delimiter=",")

	# use the convolutional autoencoder to make predictions on the
	# testing images, then initialize our list of output images
	print("[INFO] making predictions...")
	test_decoded = autoencoder.predict(test_data)
	test_outputs = None

	train_decoded = autoencoder.predict(train_data)
	train_outputs = None

	samples = len(test_data) if len(test_data) < 6 else 6

	# loop over our number of output samples
	for i in range(0, samples):

		# grab the original image and reconstructed image
		test_original = (test_data[i] * 255).astype("uint8")
		test_recon = (test_decoded[i] * 255).astype("uint8")

		train_original = (train_data[i] * 255).astype("uint8")
		train_recon = (train_decoded[i] * 255).astype("uint8")

		# stack the original and reconstructed image side-by-side
		test_output = np.hstack([test_original, test_recon])
		train_output = np.hstack([train_original, train_recon])

		# if the outputs array is empty, initialize it as the current
		# side-by-side image display
		if test_outputs is None:
			test_outputs = test_output
		# otherwise, vertically stack the outputs
		else:
			test_outputs = np.vstack([test_outputs, test_output])

		if train_outputs is None:
			train_outputs = train_output
		else:
			train_outputs = np.vstack([train_outputs, train_output])

	# save the outputs image to disk
	cv2.imwrite(decoder_plot_test, test_outputs)
	cv2.imwrite(decoder_plot_train, train_outputs)

	if save:
		full_name = "_".join(decoder_plot_name.split("_")[1:])
		ModelLoader.save_model(autoencoder.encoder, f"encoder_{full_name}.json", f"encoder_weights_{full_name}.h5")
		ModelLoader.save_model(autoencoder.decoder, f"decoder_{full_name}.json", f"decoder_weights_{full_name}.h5")

	return avg


def save_architecture(autoencoder, output_folder, filename):
	with open(f'{output_folder}/{filename}', 'w') as f:
		autoencoder.encoder.summary(print_fn=lambda x: f.write(x + '\n'))
		autoencoder.decoder.summary(print_fn=lambda x: f.write(x + '\n'))


def create_dir(output_folder):
	if os.path.isdir(output_folder):
		print(f'[ERROR]: {output_folder} already exists')
		return 1

	os.mkdir(output_folder)


def run_latent_experiment():
	epochs = 75
	bs = 8
	latents = [3, 9, 16, 32, 64, 128]

	output_folder = "../results/test_latents"
	create_dir(output_folder)

	for lat in latents:
		cae = Autoencoder().set_depth(1).set_latent(lat).build_model()
		save_architecture(cae, output_folder, f'arch_lat_{lat}.txt')
		loss = f'loss_ep_{epochs}_lat_{lat}_bs_{bs}_full'
		decoder = f'decoder_ep_{epochs}_lat_{lat}_bs_{bs}_full'
		val_loss_avg = train_autoencoder(
			epochs=epochs,
			batch_size=bs,
			autoencoder=cae,
			loss_plot_name=loss,
			decoder_plot_name=decoder,
			output_folder=output_folder,
			save=False # we do not want to save experimental models
		)
		print(f'Finished iteration for latent {lat} with {val_loss_avg} avg. test loss')
	print("Finished experiment")


def run_architecture_experiment():
	epochs = 75
	bs = 8
	kernels = [
		[(3, 3), (5, 5), (7, 7)],
		[(3, 3), (5, 5), (7, 7), (7, 7)],
		[(3, 3), (5, 5), (7, 7), (7, 7), (7, 7)]
	]
	filters = [
		[16, 32, 64],
		[16, 32, 64, 128],
		[16, 32, 64, 128, 256]
	]
	output_folder = "../results/test_architectures"
	create_dir(output_folder)

	for k, f in zip(kernels, filters):
		cae = Autoencoder().set_depth(1).set_kernel(k).set_filters(f).build_model()
		save_architecture(cae, output_folder, f'arch_arch_kf_len_{len(k)}.txt')
		loss = f'loss_ep_{epochs}_kf_len_{len(k)}_bs_{bs}_full'
		decoder = f'decoder_ep_{epochs}_kf_len_{len(k)}_{bs}_full'
		val_loss_avg = train_autoencoder(
			epochs=epochs,
			batch_size=bs,
			autoencoder=cae,
			loss_plot_name=loss,
			decoder_plot_name=decoder,
			output_folder=output_folder,
			save=False # we do not want to save experimental models
		)
		print(f'Finished iteration for arch with len {len(k)} with {val_loss_avg} avg. test loss')
	print("Finished experiment")


def run_adam_experiment():
	epochs = 75
	bs = 8
	adam_learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]

	output_folder = "../results/test_adam_lr"
	create_dir(output_folder)

	for lr in adam_learning_rates:
		cae = Autoencoder().set_depth(1).set_adam_learning_rate(lr).build_model()
		save_architecture(cae, output_folder, f'arch_lr_{lr}.txt')
		loss = f'loss_ep_{epochs}_lr_{lr}_bs_{bs}_full'
		decoder = f'decoder_ep_{epochs}_lr_{lr}_bs_{bs}_full'
		val_loss_avg = train_autoencoder(
			epochs=epochs,
			batch_size=bs,
			autoencoder=cae,
			loss_plot_name=loss,
			decoder_plot_name=decoder,
			output_folder=output_folder,
			save=False # we do not want to save experimental models
		)
		print(f'Finished iteration for learning rate {lr} with {val_loss_avg} avg. test loss')
	print("Finished experiment")


def run_leaky_relu_alpha_experiment():
	epochs = 75
	bs = 8
	leaky_relu_alphas = [0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.3]

	output_folder = "../results/test_leaky_relu_alpha"
	create_dir(output_folder)

	for val in leaky_relu_alphas:
		cae = Autoencoder().set_depth(1).set_leaky_relu_alpha(val).build_model()
		save_architecture(cae, output_folder, f'arch_relu_alpha_{val}.txt')
		loss = f'loss_ep_{epochs}_relu_alpha_{val}_bs_{bs}_full'
		decoder = f'decoder_ep_{epochs}_relu_alpha_{val}_bs_{bs}_full'
		val_loss_avg = train_autoencoder(
			epochs=epochs,
			batch_size=bs,
			autoencoder=cae,
			loss_plot_name=loss,
			decoder_plot_name=decoder,
			output_folder=output_folder,
			save=False # we do not want to save experimental models
		)
		print(f'Finished iteration for leaky relu alpha {val} with {val_loss_avg} avg. test loss')
	print("Finished experiment")


def run_batch_size_experiment():
	epochs = 75
	batch_sizes = [4, 8, 12, 32]

	output_folder = "../results/test_batch_size"
	create_dir(output_folder)

	for bs in batch_sizes:
		cae = Autoencoder().set_depth(1).build_model()
		save_architecture(cae, output_folder, f'arch_bs_{bs}.txt')
		loss = f'loss_ep_{epochs}_bs_{bs}_full'
		decoder = f'decoder_ep_{epochs}_bs_{bs}_full'
		val_loss_avg = train_autoencoder(
			epochs=epochs,
			batch_size=bs,
			autoencoder=cae,
			loss_plot_name=loss,
			decoder_plot_name=decoder,
			output_folder=output_folder,
			save=False # we do not want to save experimental models
		)
		print(f'Finished iteration for batch size {bs} with {val_loss_avg} avg. test loss')
	print("Finished experiment")


def run_best_combinations_experiment():
	epochs = 200
	bs = 32

	leaky_relu_alphas = [1.3]
	latents = [3, 64, 128]
	adam_lrs = [0.00001, 0.001, 0.01]

	output_folder = "../results/test_best_combinations"
	create_dir(output_folder)

	for lat in latents:
		for alpha in leaky_relu_alphas:
			for lr in adam_lrs:
				cae = Autoencoder()\
					.set_depth(1)\
					.set_latent(lat)\
					.set_leaky_relu_alpha(alpha)\
					.set_adam_learning_rate(lr)\
					.build_model()

				save_architecture(cae, output_folder, f'arch_bs_{bs}_lat_{lat}_relu_alpha_{alpha}_adam_lr_{lr}.txt')
				loss = f'loss_ep_{epochs}_bs_{bs}_lat_{lat}_relu_alpha_{alpha}_adam_lr_{lr}'
				decoder = f'decoder_ep_{epochs}_bs_{bs}_lat_{lat}_relu_alpha_{alpha}_adam_lr_{lr}'
				val_loss_avg = train_autoencoder(
					epochs=epochs,
					batch_size=bs,
					autoencoder=cae,
					loss_plot_name=loss,
					decoder_plot_name=decoder,
					output_folder=output_folder,
					save=False  # we do not want to save experimental models
				)

				print(f'Finished iteration for lat {lat}, alpha {alpha}, lr {lr} with {val_loss_avg} avg. test loss')
			print("Finished experiment")


def run_big_epoch_experiment():
	epochs = 1300
	bs = 32

	leaky_relu_alphas = [0.9, 1.3]
	latents = [64, 128]
	adam_lr = 0.00001

	output_folder = "../results/test_1300_epochs"
	create_dir(output_folder)

	for lat in latents:
		for alpha in leaky_relu_alphas:

			cae = Autoencoder()\
				.set_depth(1)\
				.set_latent(lat)\
				.set_leaky_relu_alpha(alpha)\
				.set_adam_learning_rate(adam_lr)\
				.build_model()

			save_architecture(cae, output_folder, f'arch_bs_{bs}_lat_{lat}_relu_alpha_{alpha}_adam_lr_{adam_lr}.txt')
			loss = f'loss_ep_{epochs}_bs_{bs}_lat_{lat}_relu_alpha_{alpha}_adam_lr_{adam_lr}'
			decoder = f'decoder_ep_{epochs}_bs_{bs}_lat_{lat}_relu_alpha_{alpha}_adam_lr_{adam_lr}'
			val_loss_avg = train_autoencoder(
				epochs=epochs,
				batch_size=bs,
				autoencoder=cae,
				loss_plot_name=loss,
				decoder_plot_name=decoder,
				output_folder=output_folder,
				save=True  # we do not want to save experimental models
			)

			print(f'Finished iteration for lat {lat}, alpha {alpha} with {val_loss_avg} avg. test loss')

		print("Finished experiment")


if __name__ == '__main__':
	# run_latent_experiment()
	# run_architecture_experiment()
	# run_adam_experiment()
	# run_batch_size_experiment()
	# run_leaky_relu_alpha_experiment()
	# run_big_epoch_experiment()

	print("Choose which experiment you would like to run")
	print("Alternatively, you can just train a CAE and use it.")
	print("Remember to choose save=True to actually save the model")
