using System;
using System.Threading.Tasks;
namespace apiDigest;

class Program
{
        static async Task Main(string[] args)
    {
        Console.WriteLine("=== API Digest ===");

            Console.WriteLine("\n-> Ejecutando Ollama...");
            await Ollama.Run(); 

            Console.WriteLine("\n-> Ejecutando Classifier...");
            await Classifier.Run();
    }
}