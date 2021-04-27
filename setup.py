from setuptools import setup, find_namespace_packages

setup(name='tuframework',
      packages=find_namespace_packages(include=["tuframework", "tuframework.*"]),
      version='1.6.6',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation.',
      url='https://github.com/MIC-DKFZ/tuframework',
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
              'tuframework_convert_decathlon_task = tuframework.experiment_planning.tuframework_convert_decathlon_task:main',
              'tuframework_plan_and_preprocess = tuframework.experiment_planning.tuframework_plan_and_preprocess:main',
              'tuframework_train = tuframework.run.run_training:main',
              'tuframework_train_DP = tuframework.run.run_training_DP:main',
              'tuframework_train_DDP = tuframework.run.run_training_DDP:main',
              'tuframework_predict = tuframework.inference.predict_simple:main',
              'tuframework_ensemble = tuframework.inference.ensemble_predictions:main',
              'tuframework_find_best_configuration = tuframework.evaluation.model_selection.figure_out_what_to_submit:main',
              'tuframework_print_available_pretrained_models = tuframework.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'tuframework_print_pretrained_model_info = tuframework.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'tuframework_download_pretrained_model = tuframework.inference.pretrained_models.download_pretrained_model:download_by_name',
              'tuframework_download_pretrained_model_by_url = tuframework.inference.pretrained_models.download_pretrained_model:download_by_url',
              'tuframework_determine_postprocessing = tuframework.postprocessing.consolidate_postprocessing_simple:main',
              'tuframework_export_model_to_zip = tuframework.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'tuframework_install_pretrained_model_from_zip = tuframework.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'tuframework_change_trainer_class = tuframework.inference.change_trainer:main',
              'tuframework_evaluate_folder = tuframework.evaluation.evaluator:tuframework_evaluate_folder'
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'tuframework']
      )
