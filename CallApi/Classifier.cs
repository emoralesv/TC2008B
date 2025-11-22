
namespace apiDigest;

using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class Classifier
{
   public static async Task Run()
    {
        try
        {
            // Ruta de la imagen a enviar al servidor FastAPI
            string imagePath = @"C:\Users\eduar\repos\TC2008B\TestImages\T_F_H\00042_jpg.rf.08b5afd1b38abb655bde467761e7e009_T_F_H_001.jpg";

            // URL del endpoint FastAPI que recibe la imagen
            string url = "http://127.0.0.1:80/predict";

            // Cliente HTTP (permite enviar peticiones POST, GET, etc.)
            using var client = new HttpClient();

            // Contenedor especial para enviar archivos
            using var content = new MultipartFormDataContent();

            // Leer los bytes de la imagen desde disco
            var imageBytes = File.ReadAllBytes(imagePath);

            // Crear contenido binario con los bytes de la imagen
            var imageContent = new ByteArrayContent(imageBytes);

            // Definir el tipo de archivo (obligatorio para FastAPI)
            imageContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");

            // Añadir la imagen al formulario
            content.Add(imageContent, "file", Path.GetFileName(imagePath));

            // Enviar solicitud POST
            var response = await client.PostAsync(url, content);

            // Lanzar excepción si FastAPI devolvió error (4xx o 5xx)
            response.EnsureSuccessStatusCode();

            // Leer respuesta del servidor
            var responseString = await response.Content.ReadAsStringAsync();

            Console.WriteLine("Response:");
            Console.WriteLine(responseString);
        }
        catch (FileNotFoundException ex)
        {
            Console.WriteLine($"No se encontró la imagen: {ex.Message}");
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine($"Error al enviar la solicitud HTTP: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error inesperado: {ex.Message}");
        }
    }
}