List of all programs and their purposes

syllable_sorting.ipynb
automated_umap_syllables.ipynb
umap_syllables.ipynb
song_spikes.ipynb

analysis_pipeline.ipynb
umapping_songs.ipynb

# Feature Extraction Pipeline #
Features based on Sound Analysis Pro (2011) and Librosa
    * SAP_features.ipynb
        Preliminary implementation of SAP features in a notebook:
            STFT and Spectral Derivative (Primitive T/F Derivative)
            Amplitude (log based amplitude)
            Pitch Estimation (Using Peak/Mean/Yin Methods)
            Wiener entropy (normalized)
            Frequency Modulation
            Amplitude Modulation
            Spectral Continuity (Weird and parameter dependent)
            Spectral Contrast (From Librosa)
            Mel Spectrogram (Compared to Regular Spectrogram/Spectrogram Derivative)

    * SAP_features.py
        Python script for features validated in SAP_features.ipynb

    * SAP_features_test.ipynb
        Testing notebook for SAP_features.py

    * feature_imp.ipynb
        Combining all features into one notebook, copying the implementation from SAP and adding librosa features

# Using Features to Separate Syllables #
    * syllable_sorting.ipynb
        Segment syllables according to spectral entropy, and store as a pkl file

# Feature Embedding # 
    * automated_umap_syllables.ipynb
        Get segmented syllables from previous code, extract features, and plot them onto UMAP embedding

    * umap_syllables.ipynb
        Use hand-segmented data with spectral features to get umap embedding

# Syllable Analysis #
    * song_spikes.ipynb
        Look for patterns and connections between syllables (preliminary)

# Tools #
    * videoFormatterBirdsong.ipynb
        Combine all recording data for one day into a single video, renaming all appropriate files

    * time_alignment.ipynb
        First attempt at aligning neural data to the audio recordings





