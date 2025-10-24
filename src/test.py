import os

from module.svd_face_recognition import SVDFaceRecognition

def main():
    sample_dir = "data"
    training_data = {}
    for person_dir in os.listdir(sample_dir):
        person_path = os.path.join(sample_dir, person_dir)
        if os.path.isdir(person_path):
            image_paths = []
            for file in os.listdir(person_path):
                if file.lower().endswith(('.jpg')):
                    image_paths.append(os.path.join(person_path, file))
            if image_paths:
                training_data[person_dir] = image_paths
    
    # Crear y entrenar modelo
    model = SVDFaceRecognition(epsilon_1=0.3, epsilon_0=0.2)
    model.train(training_data)
    
    # Probar reconocimiento
    print("\n=== PROBANDO RECONOCIMIENTO ===")
    for person, paths in training_data.items():
        if paths:
            test_image = paths[0]  # Primera imagen de cada persona
            recognized, distance, all_distances = model.recognize_face(test_image)
            
            print(f"\nImagen: {os.path.basename(test_image)}")
            print(f"Persona reconocida: {recognized}")
            print(f"Distancia mínima: {distance:.4f}")
            print(f"Distancias a todas las personas:")
            for p, d in all_distances.items():
                print(f"  {p}: {d:.4f}")
    

if __name__ == "__main__":
    main()