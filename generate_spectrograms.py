"""
Script pour générer des spectres de fréquence pour tous les fichiers audio dans audio/filters/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path


def generate_amplitude_spectrum(audio_path, output_path):
    """
    Génère et sauvegarde un spectre d'amplitude normalisé (amplitude entre 0 et 1 par fréquence)
    
    Args:
        audio_path: Chemin vers le fichier audio
        output_path: Chemin de sortie pour l'image du spectre
    """
    try:
        # Lire le fichier audio
        sample_rate, data = wavfile.read(audio_path)
        
        # Convertir en mono si stéréo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Normaliser les données
        data = data.astype(float)
        
        # Calculer la FFT
        fft = np.fft.fft(data)
        fft_magnitude = np.abs(fft)
        fft_freq = np.fft.fftfreq(len(data), 1/sample_rate)
        
        # Ne garder que les fréquences positives
        positive_freqs = fft_freq[:len(fft_freq)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        
        # Normaliser l'amplitude entre 0 et 1
        amplitude_normalized = positive_magnitude / (np.max(positive_magnitude) + 1e-10)
        
        # Créer la figure
        plt.figure(figsize=(14, 7))
        plt.plot(positive_freqs, amplitude_normalized, linewidth=1.2, color='#A23B72')
        
        # Labels et titre
        plt.xlabel('Fréquence (Hz)', fontsize=12, fontweight='bold')
        plt.ylabel('Amplitude normalisée (0-1)', fontsize=12, fontweight='bold')
        plt.title(f'Amplitude par fréquence - {Path(audio_path).name}', fontsize=14, fontweight='bold', pad=20)
        
        # Limites
        freq_max = min(sample_rate/2, 20000)
        plt.xlim(20, freq_max)  # Commencer à 20 Hz
        plt.ylim(0, 1.05)  # Plage de 0 à 1
        
        # Échelle logarithmique pour l'axe X avec ticks personnalisés
        plt.xscale('log')
        
        # Ticks personnalisés pour l'axe X
        x_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        x_ticks = [x for x in x_ticks if x <= freq_max]
        plt.xticks(x_ticks, [f'{x}' if x < 1000 else f'{x//1000}k' for x in x_ticks])
        
        # Grille améliorée
        plt.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.8)
        plt.grid(True, which='minor', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Style supplémentaire
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Sauvegarder avec meilleure qualité
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Spectre d'amplitude créé: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Erreur pour {audio_path}: {e}")
        return False


def generate_spectrum(audio_path, output_path):
    """
    Génère et sauvegarde le spectre de fréquence d'un fichier audio
    
    Args:
        audio_path: Chemin vers le fichier audio
        output_path: Chemin de sortie pour l'image du spectre
    """
    try:
        # Lire le fichier audio
        sample_rate, data = wavfile.read(audio_path)
        
        # Convertir en mono si stéréo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Normaliser les données
        data = data.astype(float)
        
        # Calculer la FFT
        fft = np.fft.fft(data)
        fft_magnitude = np.abs(fft)
        fft_freq = np.fft.fftfreq(len(data), 1/sample_rate)
        
        # Ne garder que les fréquences positives
        positive_freqs = fft_freq[:len(fft_freq)//2]
        positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
        
        # Convertir en dB et normaliser
        magnitude_db = 20 * np.log10(positive_magnitude + 1e-10)
        magnitude_db = magnitude_db - np.max(magnitude_db)  # Normaliser au max à 0 dB
        
        # Créer la figure
        plt.figure(figsize=(14, 7))
        plt.plot(positive_freqs, magnitude_db, linewidth=1.2, color='#2E86AB')
        
        # Labels et titre
        plt.xlabel('Fréquence (Hz)', fontsize=12, fontweight='bold')
        plt.ylabel('Magnitude (dB)', fontsize=12, fontweight='bold')
        plt.title(f'Spectre de fréquence - {Path(audio_path).name}', fontsize=14, fontweight='bold', pad=20)
        
        # Limites
        freq_max = min(sample_rate/2, 20000)
        plt.xlim(20, freq_max)  # Commencer à 20 Hz
        plt.ylim(-80, 5)  # Plage de -80 à +5 dB
        
        # Échelle logarithmique pour l'axe X avec ticks personnalisés
        plt.xscale('log')
        
        # Ticks personnalisés pour l'axe X
        x_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        x_ticks = [x for x in x_ticks if x <= freq_max]
        plt.xticks(x_ticks, [f'{x}' if x < 1000 else f'{x//1000}k' for x in x_ticks])
        
        # Grille améliorée
        plt.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.8)
        plt.grid(True, which='minor', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # Style supplémentaire
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Sauvegarder avec meilleure qualité
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Spectre créé: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Erreur pour {audio_path}: {e}")
        return False


def generate_spectrogram(audio_path, output_path):
    """
    Génère et sauvegarde un spectrogramme (temps-fréquence)
    
    Args:
        audio_path: Chemin vers le fichier audio
        output_path: Chemin de sortie pour l'image du spectrogramme
    """
    try:
        from scipy import signal
        
        # Lire le fichier audio
        sample_rate, data = wavfile.read(audio_path)
        
        # Convertir en mono si stéréo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Normaliser les données
        data = data.astype(float)
        
        # Calculer le spectrogramme
        frequencies, times, Sxx = signal.spectrogram(
            data, 
            fs=sample_rate,
            window='hann',
            nperseg=2048,
            noverlap=1536,
            scaling='spectrum'
        )
        
        # Convertir en dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Afficher le spectrogramme
        pcm = ax.pcolormesh(
            times, 
            frequencies, 
            Sxx_db,
            shading='gouraud',
            cmap='viridis',
            vmin=np.percentile(Sxx_db, 5),
            vmax=np.percentile(Sxx_db, 95)
        )
        
        # Ajouter une barre de couleur
        cbar = plt.colorbar(pcm, ax=ax, label='Amplitude (dB)')
        cbar.ax.tick_params(labelsize=10)
        
        # Labels et titre
        ax.set_xlabel('Temps (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fréquence (Hz)', fontsize=12, fontweight='bold')
        ax.set_title(f'Spectrogramme - {Path(audio_path).name}', fontsize=14, fontweight='bold', pad=20)
        
        # Échelle logarithmique pour les fréquences
        ax.set_yscale('log')
        ax.set_ylim(20, min(20000, sample_rate/2))
        
        # Ticks personnalisés pour l'axe Y
        y_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        y_ticks = [y for y in y_ticks if y <= sample_rate/2]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y}' if y < 1000 else f'{y//1000}k' for y in y_ticks])
        
        # Grille
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Sauvegarder avec meilleure qualité
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Spectrogramme créé: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Erreur pour {audio_path}: {e}")
        return False


def generate_oscilloscope_first_cycle(audio_path, output_path):
    """
    Génère un oscilloscope des 3 premières oscillations (entre 4 traversées de zéro).
    Si moins d'oscillations sont trouvées, on trace le début du signal (jusqu'à 4096 échantillons).
    """
    try:
        sample_rate, data = wavfile.read(audio_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        data = data.astype(float)
        if data.size == 0:
            raise ValueError("signal vide")

        sign_bits = np.signbit(data)
        zero_crossings = np.flatnonzero(np.diff(sign_bits))

        if zero_crossings.size >= 4:
            start = zero_crossings[0]
            end = zero_crossings[3] + 1
            segment = data[start:end]
        else:
            start = 0
            end = min(len(data), 4096)
            segment = data[start:end]

        if segment.size == 0:
            raise ValueError("segment vide pour oscilloscope")

        t = np.arange(segment.size) / float(sample_rate)

        plt.figure(figsize=(12, 4))
        plt.plot(t, segment, linewidth=1.0, color="#D1495B")
        plt.xlabel("Temps (s)", fontsize=11, fontweight="bold")
        plt.ylabel("Amplitude", fontsize=11, fontweight="bold")
        plt.title(f"Oscilloscope 3 premières oscillations - {Path(audio_path).name}\n(samples {start}→{start+segment.size})",
                  fontsize=12, fontweight="bold", pad=14)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✓ Oscilloscope créé: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Erreur oscilloscope pour {audio_path}: {e}")
        return False


def main():
    """Fonction principale"""
    # Dossier de base
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "audio"
    filters_dir = audio_dir / "filters"
    comb_dir = audio_dir / "comb"
    softclip_dir = audio_dir / "soft-clip"
    hardclip_dir = audio_dir / "hard-clip"
    spectra_dir = base_dir / "spectra"
    
    # Créer le dossier de sortie
    spectra_dir.mkdir(exist_ok=True, parents=True)
    
    # Fichiers pour spectres/amplitudes (filters + comb)
    filters_files = list(filters_dir.rglob("*.wav")) if filters_dir.exists() else []
    comb_files = list(comb_dir.glob("*.wav")) if comb_dir.exists() else []
    handled_files = filters_files + comb_files

    # Fichiers de distorsion pour oscilloscopes (soft/hard clip)
    softclip_files = list(softclip_dir.rglob("*.wav")) if softclip_dir.exists() else []
    hardclip_files = list(hardclip_dir.rglob("*.wav")) if hardclip_dir.exists() else []
    distortion_files = softclip_files + hardclip_files
    distortion_files_no_melody = [f for f in distortion_files if 'melody' not in f.stem.lower()]
    # Filtrer pour exclure les fichiers melody (pour spectres et amplitudes uniquement)
    handled_files_no_melody = [f for f in handled_files if 'melody' not in f.stem.lower()]
    
    # Tous les fichiers audio (spectrogrammes y compris melody)
    all_audio_files = list(audio_dir.rglob("*.wav"))
    
    print(f"Fichiers trouvés:")
    print(f"  - Total audio: {len(all_audio_files)}")
    print(f"  - Filters: {len(filters_files)}")
    print(f"  - Comb: {len(comb_files)}")
    print(f"  - Soft-clip: {len(softclip_files)}")
    print(f"  - Hard-clip: {len(hardclip_files)}")
    print(f"\nFichiers melody ignorés (spectres & amplitudes & oscilloscopes): {len(handled_files) - len(handled_files_no_melody) + len(distortion_files) - len(distortion_files_no_melody)}")
    print(f"\nFichiers à traiter:")
    print(f"  - Spectrogrammes (tous): {len(all_audio_files)}")
    print(f"  - Spectres & Amplitudes (filters + comb, sans melody): {len(handled_files_no_melody)}")
    print(f"  - Oscilloscopes (soft/hard clip, sans melody): {len(distortion_files_no_melody)}")
    print(f"Génération dans: {spectra_dir}\n")
    
    success_spectrum_count = 0
    success_amplitude_count = 0
    success_spectrogram_count = 0
    success_oscilloscope_count = 0
    skipped_spectrum = 0
    skipped_amplitude = 0
    skipped_spectrogram = 0
    skipped_oscilloscope = 0
    error_count = 0
    
    # Générer spectres et amplitudes UNIQUEMENT pour filters et comb
    for wav_file in handled_files_no_melody:
        # Créer le chemin de sortie en conservant la structure
        relative_path = wav_file.relative_to(audio_dir)
        output_subdir = spectra_dir / relative_path.parent
        output_subdir.mkdir(exist_ok=True, parents=True)
        
        # Générer le spectre de fréquence (dB)
        output_spectrum = output_subdir / f"{wav_file.stem}_spectrum.png"
        if output_spectrum.exists():
            skipped_spectrum += 1
        else:
            if generate_spectrum(str(wav_file), str(output_spectrum)):
                success_spectrum_count += 1
            else:
                error_count += 1
        
        # Générer le spectre d'amplitude (0-1)
        output_amplitude = output_subdir / f"{wav_file.stem}_amplitude.png"
        if output_amplitude.exists():
            skipped_amplitude += 1
        else:
            if generate_amplitude_spectrum(str(wav_file), str(output_amplitude)):
                success_amplitude_count += 1
            else:
                error_count += 1

    # Générer oscilloscopes pour les distorsions (soft/hard clip)
    for wav_file in distortion_files_no_melody:
        relative_path = wav_file.relative_to(audio_dir)
        output_subdir = spectra_dir / relative_path.parent
        output_subdir.mkdir(exist_ok=True, parents=True)

        output_osc = output_subdir / f"{wav_file.stem}_oscilloscope.png"
        if output_osc.exists():
            skipped_oscilloscope += 1
        else:
            if generate_oscilloscope_first_cycle(str(wav_file), str(output_osc)):
                success_oscilloscope_count += 1
            else:
                error_count += 1
    
    # Générer spectrogrammes pour TOUS les fichiers audio (inclut melody)
    for wav_file in all_audio_files:
        # Créer le chemin de sortie en conservant la structure
        relative_path = wav_file.relative_to(audio_dir)
        output_subdir = spectra_dir / relative_path.parent
        output_subdir.mkdir(exist_ok=True, parents=True)
        
        # Générer le spectrogramme
        output_spectrogram = output_subdir / f"{wav_file.stem}_spectrogram.png"
        if output_spectrogram.exists():
            skipped_spectrogram += 1
        else:
            if generate_spectrogram(str(wav_file), str(output_spectrogram)):
                success_spectrogram_count += 1
            else:
                error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Terminé!")
    print(f"Spectres (dB) créés: {success_spectrum_count} (ignorés: {skipped_spectrum})")
    print(f"Spectres d'amplitude (0-1) créés: {success_amplitude_count} (ignorés: {skipped_amplitude})")
    print(f"Oscilloscopes (première oscillation) créés: {success_oscilloscope_count} (ignorés: {skipped_oscilloscope})")
    print(f"Spectrogrammes créés: {success_spectrogram_count} (ignorés: {skipped_spectrogram})")
    print(f"Erreurs: {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
