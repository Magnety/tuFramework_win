from setuptools import setup, find_namespace_packages

setup(name='tuframework',
      packages=find_namespace_packages(include=["tuframework", "tuframework.*"]),
      version='1.6.6',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation.',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'nnUNet_convert_decathlon_task = tuframework.experiment_planning.nnUNet_convert_decathlon_task:main',
              'nnUNet_plan_and_preprocess = tuframework.experiment_planning.nnUNet_plan_and_preprocess:main',
              'nnUNet_train = tuframework.run.run_training:main',
              'nnUNet_train_DP = tuframework.run.run_training_DP:main',
              'nnUNet_train_DDP = tuframework.run.run_training_DDP:main',
              'nnUNet_predict = tuframework.inference.predict_simple:main',
              'nnUNet_ensemble = tuframework.inference.ensemble_predictions:main',
              'nnUNet_find_best_configuration = tuframework.evaluation.model_selection.figure_out_what_to_submit:main',
              'nnUNet_print_available_pretrained_models = tuframework.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'nnUNet_print_pretrained_model_info = tuframework.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'nnUNet_download_pretrained_model = tuframework.inference.pretrained_models.download_pretrained_model:download_by_name',
              'nnUNet_download_pretrained_model_by_url = tuframework.inference.pretrained_models.download_pretrained_model:download_by_url',
              'nnUNet_determine_postprocessing = tuframework.postprocessing.consolidate_postprocessing_simple:main',
              'nnUNet_export_model_to_zip = tuframework.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'nnUNet_install_pretrained_model_from_zip = tuframework.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'nnUNet_change_trainer_class = tuframework.inference.change_trainer:main',
              'nnUNet_evaluate_folder = tuframework.evaluation.evaluator:tuframework_evaluate_folder'
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'tuframework']
      )
