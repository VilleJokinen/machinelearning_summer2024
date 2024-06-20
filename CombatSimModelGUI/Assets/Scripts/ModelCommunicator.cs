using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

public class ModelCommunicator : MonoBehaviour
{
    public string serverUrl = "http://localhost:5000/predict";

    public IEnumerator PostRequest(string url, string jsonData, System.Action<int> callback)
    {
        UnityWebRequest request = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Response: " + request.downloadHandler.text);
            var response = JsonUtility.FromJson<PredictionResponse>(request.downloadHandler.text);
            callback?.Invoke(response.result);
        }
        else
        {
            Debug.Log("Error: " + request.error);
        }
    }

    public void SendDataToModel(int[] p1Cards, int[] p2Cards, System.Action<int> callback)
    {
        PredictionRequest data = new PredictionRequest { p1_cards = p1Cards, p2_cards = p2Cards };
        string jsonData = JsonUtility.ToJson(data);
        StartCoroutine(PostRequest(serverUrl, jsonData, callback));
    }

    [System.Serializable]
    public class PredictionRequest
    {
        public int[] p1_cards;
        public int[] p2_cards;
    }

    [System.Serializable]
    public class PredictionResponse
    {
        public int result;
    }

}
