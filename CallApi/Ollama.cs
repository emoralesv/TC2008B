namespace apiDigest;
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;

class Ollama
{
    public static async Task Run()
    {   
        // Cliente HTTP (permite enviar peticiones POST, GET, etc.)
        using var client = new HttpClient();

        // URL del servicio
        var url = "http://localhost:11434/api/generate";

        // JSON de la solicitud
        string json = GenerateJson(
            model: "llama3",                            // Modelo LLM
            prompt: "what is the capital of France",    // Consulta/promp de usuario
            system: "",                                 // Promp del sistema. Cómo debe actuar el LLM.
            stream: false                               // Tipo de respuesta: true, false.
        );

        // Configurar la solicitud
        var request = new HttpRequestMessage(HttpMethod.Post, url);
        request.Content = new StringContent(json, Encoding.UTF8, "application/json");

        try
        {
            // Enviar la solicitud
            // Si proceso continua con la ejecución del programa principal, una vez que reciba la respuesta continuará
            var response = await client.SendAsync(request);
            // Lanza una excepción (error) si la respuesta fue un error (ejecuta catch)
            response.EnsureSuccessStatusCode();

            // Leer la respuesta
            var responseBody = await response.Content.ReadAsStringAsync();
            Console.WriteLine("Respuesta del servidor:");
            Console.WriteLine(responseBody);
        }
        catch (HttpRequestException e)
        {
            Console.WriteLine($"Error al enviar la solicitud: {e.Message}");
        }
    }
    public static string GenerateJson(string model, string prompt, string system, bool stream)
    {
        return $@"{{
    ""model"": ""{model}"",
    ""prompt"": ""{prompt}"",
    ""system"": ""{system}"",
    ""stream"": {stream.ToString().ToLower()}
    }}";
    }
}
