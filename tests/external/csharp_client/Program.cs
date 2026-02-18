using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace HanteiApiClient
{
    class Program
    {
        private static readonly HttpClient client = new HttpClient();
        private const string BASE_URL = "http://localhost:8000";

        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.WriteLine("Hantei API Test Client\n");

            // 1. Health Check
            await TestHealthCheck();

            // 2. Predict (Single Image)
            // テスト用のダミー画像を作成 (1x1 pixel white jpg)
            string dummyImageBase64 = CreateDummyImageBase64();
            await TestPredictSingle(dummyImageBase64);

            // 3. Batch Predict
            await TestPredictBatch(dummyImageBase64);

            // 4. Get Categories
            await TestGetCategories();

            Console.WriteLine("\nTests completed. Press any key to exit...");
            Console.ReadKey();
        }

        static async Task TestHealthCheck()
        {
            Console.WriteLine("--- Testing /health ---");
            try
            {
                HttpResponseMessage response = await client.GetAsync($"{BASE_URL}/health");
                response.EnsureSuccessStatusCode();
                string responseBody = await response.Content.ReadAsStringAsync();
                Console.WriteLine($"Response: {responseBody}");
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
            Console.WriteLine();
        }

        static async Task TestPredictSingle(string imageBase64)
        {
            Console.WriteLine("--- Testing /api/v1/predict ---");
            try
            {
                var requestData = new
                {
                    image_base64 = imageBase64,
                    return_confidence = true
                };

                string json = JsonConvert.SerializeObject(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                HttpResponseMessage response = await client.PostAsync($"{BASE_URL}/api/v1/predict", content);
                
                if (response.IsSuccessStatusCode)
                {
                    string responseBody = await response.Content.ReadAsStringAsync();
                    // 整形して表示
                    var parsedJson = JToken.Parse(responseBody);
                    Console.WriteLine($"Response: {parsedJson.ToString(Formatting.Indented)}");
                }
                else
                {
                    Console.WriteLine($"Error: {response.StatusCode}");
                    string errorBody = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"Details: {errorBody}");
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
            Console.WriteLine();
        }

        static async Task TestPredictBatch(string imageBase64)
        {
            Console.WriteLine("--- Testing /api/v1/predict/batch ---");
            try
            {
                var requestData = new
                {
                    images = new List<string> { imageBase64, imageBase64 },
                    return_confidence = true
                };

                string json = JsonConvert.SerializeObject(requestData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                HttpResponseMessage response = await client.PostAsync($"{BASE_URL}/api/v1/predict/batch", content);

                if (response.IsSuccessStatusCode)
                {
                    string responseBody = await response.Content.ReadAsStringAsync();
                    var parsedJson = JToken.Parse(responseBody);
                    Console.WriteLine($"Response: {parsedJson.ToString(Formatting.Indented)}");
                }
                else
                {
                    Console.WriteLine($"Error: {response.StatusCode}");
                    string errorBody = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"Details: {errorBody}");
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
            Console.WriteLine();
        }

        static async Task TestGetCategories()
        {
            Console.WriteLine("--- Testing /api/v1/categories ---");
            try
            {
                HttpResponseMessage response = await client.GetAsync($"{BASE_URL}/api/v1/categories");
                response.EnsureSuccessStatusCode();
                string responseBody = await response.Content.ReadAsStringAsync();
                var parsedJson = JToken.Parse(responseBody);
                Console.WriteLine($"Response: {parsedJson.ToString(Formatting.Indented)}");
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
            Console.WriteLine();
        }

        static string CreateDummyImageBase64()
        {
            // 1x1 white pixel PNG (Verified)
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";
        }
    }
}
