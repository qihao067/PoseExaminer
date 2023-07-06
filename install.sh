conda env create -f adv_env.yaml

rm -rf third_party
mkdir third_party
cd third_party

pip install git+https://github.com/mkocabas/yolov3-pytorch.git
pip install git+https://github.com/mkocabas/multi-person-tracker.git
pip install git+https://github.com/giacaglia/pytube.git

git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
python setup.py develop

pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.1.1/index.html
pip install "mmdet<=2.25.1"
pip install "mmpose<=0.28.1"

cd ../..
