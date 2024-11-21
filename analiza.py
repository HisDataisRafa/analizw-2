import streamlit as st
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import tempfile
import os

def analyze_audio(audio_path):
    """Análisis básico de audio"""
    try:
        # Leer el archivo WAV
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convertir a mono si es estéreo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalizar los datos
        audio_data = audio_data.astype(float) / np.max(np.abs(audio_data))
        
        # Calcular el espectrograma
        frequencies, times, Sxx = signal.spectrogram(audio_data, fs=sample_rate)
        
        # Encontrar frecuencias dominantes
        dominant_frequencies = []
        for time_idx in range(Sxx.shape[1]):
            freq_idx = np.argmax(Sxx[:, time_idx])
            if Sxx[freq_idx, time_idx] > 0.1:  # Umbral de energía
                dominant_frequencies.append(frequencies[freq_idx])
        
        # Análisis básico
        if dominant_frequencies:
            avg_freq = np.mean(dominant_frequencies)
            voice_type = "Masculina" if avg_freq < 150 else "Femenina"
            
            return {
                "tipo_voz": voice_type,
                "frecuencia_promedio": avg_freq,
                "audio_data": audio_data,
                "sample_rate": sample_rate
            }
            
    except Exception as e:
        st.error(f"Error en el análisis: {str(e)}")
        return None

def main():
    st.title("🎤 Analizador de Voz Básico")
    
    st.write("""
    ### Sube un archivo de audio WAV para analizar
    Nota: Por favor, asegúrate de que el archivo esté en formato WAV
    """)
    
    uploaded_file = st.file_uploader("Selecciona un archivo WAV", type=['wav'])
    
    if uploaded_file:
        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        try:
            with st.spinner("Analizando audio..."):
                results = analyze_audio(temp_path)
            
            if results:
                st.success("¡Análisis completado!")
                
                # Mostrar resultados básicos
                st.subheader("Resultados:")
                st.write(f"**Tipo de voz detectada:** {results['tipo_voz']}")
                st.write(f"**Frecuencia promedio:** {results['frecuencia_promedio']:.2f} Hz")
                
                # Crear visualización simple
                fig, ax = plt.subplots(figsize=(10, 4))
                time = np.arange(len(results['audio_data'])) / results['sample_rate']
                ax.plot(time, results['audio_data'])
                ax.set_title('Forma de Onda del Audio')
                ax.set_xlabel('Tiempo (segundos)')
                ax.set_ylabel('Amplitud')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
    
    st.markdown("""
    ### Instrucciones:
    1. Sube un archivo de audio en formato WAV
    2. Espera el análisis
    3. Revisa los resultados y la gráfica
    
    ### Notas:
    - Solo acepta archivos WAV
    - Usa grabaciones claras para mejores resultados
    - Evita ruido de fondo
    - Duración recomendada: 5-30 segundos
    """)

if __name__ == "__main__":
    main()
