# export USE_TF=0
# HYDRA_FULL_ERROR=1 python train_tokenizer.py

# export USE_TF=0
# HYDRA_FULL_ERROR=1 /Users/dmitrii/miniconda3/bin/python train_tokenizer.py

# export USE_TF=0
# HYDRA_FULL_ERROR=1 /Users/dmitrii/miniconda3/bin/python train_diffusion.py \
#     trainer.accelerator=mps \
#     trainer.precision=32 \
#     trainer.fast_dev_run=true \
#     dataloaders.train.batch_size=2 \
#     dataloaders.val.batch_size=2 \
#     dataloaders.train.num_workers=0 \
#     dataloaders.val.num_workers=0 \
#     dataloaders.train.persistent_workers=false \
#     dataloaders.val.persistent_workers=false

# export USE_TF=0
# HYDRA_FULL_ERROR=1 /Users/dmitrii/miniconda3/bin/python train_diffusion.py \
#     trainer.accelerator=mps \
#     trainer.precision=32 \
#     trainer.max_epochs=3 \
#     trainer.limit_train_batches=20 \
#     +trainer.limit_val_batches=10 \
#     dataloaders.train.batch_size=16 \
#     dataloaders.val.batch_size=16 \
#     dataloaders.train.num_workers=0 \
#     dataloaders.val.num_workers=0 \
#     dataloaders.train.persistent_workers=false \
#     dataloaders.val.persistent_workers=false

export USE_TF=0
HYDRA_FULL_ERROR=1 /Users/dmitrii/miniconda3/bin/python train_diffusion.py \
    training_pipeline=base \
    trainer.accelerator=mps \
    trainer.precision=32 \
    trainer.max_epochs=15 \
    dataloaders.train.batch_size=64 \
    dataloaders.val.batch_size=64 \
    dataloaders.train.num_workers=0 \
    dataloaders.val.num_workers=0 \
    dataloaders.train.persistent_workers=false \
    dataloaders.val.persistent_workers=false