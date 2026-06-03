"""Per-modality adapter stubs. Each module a concrete subclass of BasePerturbation
would live in (cv.py, mri.py, eeg.py); for the MVP they're documented here as
TODOs pointing at the canonical per-modality toolkit so a contributor can fill
in a concrete adapter without re-deriving the design.

CV (Kitware NRTK)
    https://github.com/Kitware/nrtk -- physics-based sensor perturbations
    (focal length, aperture, pixel pitch, sensor noise) via pyBSM. The right
    adapter wraps nrtk.impls.perturb_image.* into the BasePerturbation
    interface; severity 1.0 should map to a documented "operationally
    relevant maximum" per nrtk's published parameter ranges.

MRI (TorchIO)
    https://torchio.readthedocs.io -- domain-realistic MRI perturbations
    (motion, susceptibility distortion, bias field, ghosting, spike). Wrap
    torchio.RandomMotion, RandomBiasField, RandomGhosting, RandomSpike as
    severity-parameterized BasePerturbations. Used by smri-fm and any
    realtime-mindeye experiment.

EEG (MNE-Python)
    https://mne.tools -- electrode dropout (mne.Epochs.copy().drop_channels),
    line noise (mne.preprocessing add_eeg_artifact-style), blink/EMG (via
    ICA-based subtraction or direct templated injection). Severity controls
    fraction of epochs affected and per-channel SNR drop.

Why these are not pinned dependencies of wwj
    Each modality toolkit has its own (substantial) dependency tree; pinning
    them as required dependencies of wwj would make a CV-only or MRI-only
    user pay for the others. Concrete adapters should live in optional-extras
    install groups (`pip install wwj[cv]`, `wwj[mri]`, `wwj[eeg]`) and each
    adapter should import the toolkit lazily inside its `apply` method.
"""
