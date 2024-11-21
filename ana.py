import streamlit as st
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tempfile
import os

def analyze_audio(audio_path):
    """Función simple para analizar audio"""
    # Cargar el archivo de audio
    audio_data, sample_rate = sf.read(audio_path)
    
    # Si el audio es estéreo, convertir a mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Calcular la frecuencia fundamental
    frame_size = 2048
    hop_length = 512
    
    # Calcular espectrograma
    frequencies, times, Sxx = signal.spectrogram(audio_data, fs=sample_rate,
                                               nperseg=frame_size,
                                               noverlap=hop_length)
    
    # Encontrar frecuencias dominantes
    dominant_freqs = []
    for time_idx in range(Sxx.shape[1]):
        if np.max(Sxx[:, time_idx]) > 0.01:  # Umbral de energía
            freq_idx = np.argmax(Sxx[:, time_idx])
            dominant_freqs.append(frequencies[freq_idx])
    
    # Calcular estadísticas
    if dominant_freqs:
        avg_freq = np.mean(dominant_freqs)
        # Clasificación simple basada en la frecuencia promedio
        voice_type = "Masculina" if avg_freq < 150 else "Femenina"
        
        return {
            "tipo_voz": voice_type,
            "frecuencia_promedio": avg_freq,
            "frecuencias": dominant_freqs,
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
    return None

def main():
    st.title("📊 Analizador de Voz Simple")
    
    st.write("""
    ### Sube un archivo de audio para analizar
    Acepta archivos WAV y MP3
    """)
    
    uploaded_file = st.file_uploader("Selecciona un archivo de audio", type=['wav', 'mp3'])
    
    if uploaded_file:
        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        try:
            # Analizar el audio
            with st.spinner("Analizando audio..."):
                results = analyze_audio(temp_path)
            
            if results:
                st.success("¡Análisis completado!")
                
                # Mostrar resultados
                st.subheader("Resultados:")
                st.write(f"**Tipo de voz detectada:** {results['tipo_voz']}")
                st.write(f"**Frecuencia promedio:** {results['frecuencia_promedio']:.2f} Hz")
                
                # Crear gráfica de forma de onda
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Forma de onda
                time = np.arange(len(results['audio_data'])) / results['sample_rate']
                ax1.plot(time, results['audio_data'])
                ax1.set_title('Forma de Onda')
                ax1.set_xlabel('Tiempo (s)')
                ax1.set_ylabel('Amplitud')
                
                # Gráfica de frecuencias
                ax2.plot(results['frecuencias'])
                ax2.set_title('Frecuencias Dominantes')
                ax2.set_ylabel('Frecuencia (Hz)')
                ax2.set_xlabel('Fragmento de Tiempo')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.error("No se pudo analizar el audio. Intenta con otro archivo.")
                
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
    
    st.markdown("""
    ### Cómo usar:
    1. Sube un archivo de audio (WAV o MP3)
    2. Espera el análisis
    3. Revisa los resultados y gráficas
    
    ### Recomendaciones:
    - Usa grabaciones claras
    - Evita ruido de fondo
    - Duración recomendada: 5-30 segundos
    """)

if __name__ == "__main__":
    main()
