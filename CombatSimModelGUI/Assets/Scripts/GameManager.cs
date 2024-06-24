using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class GameManager : MonoBehaviour
{
    public TMP_InputField p1Card1;
    public TMP_InputField p1Card2;
    public TMP_InputField p1Card3;
    public TMP_InputField p1Color1;
    public TMP_InputField p1Color2;
    public TMP_InputField p1Color3;
    public TMP_InputField p2Card1;
    public TMP_InputField p2Card2;
    public TMP_InputField p2Card3;
    public TMP_InputField p2Color1;
    public TMP_InputField p2Color2;
    public TMP_InputField p2Color3;
    public TextMeshProUGUI resultText;
    [SerializeField] private ModelCommunicator modelCommunicator;

    void Start()
    {
        modelCommunicator = GetComponent<ModelCommunicator>();
    }

    public void OnSubmit()
    {
        // Debugging output
        Debug.Log($"p1Card1: {p1Card1.text}, p1Card2: {p1Card2.text}, p1Card3: {p1Card3.text}");
        Debug.Log($"p1Color1: {p1Color1.text}, p1Color2: {p1Color2.text}, p1Color3: {p1Color3.text}");
        Debug.Log($"p2Card1: {p2Card1.text}, p2Card2: {p2Card2.text}, p2Card3: {p2Card3.text}");
        Debug.Log($"p2Color1: {p2Color1.text}, p2Color2: {p2Color2.text}, p2Color3: {p2Color3.text}");

        if (!ValidateInput(p1Card1.text) || !ValidateInput(p1Card2.text) || !ValidateInput(p1Card3.text) ||
            !ValidateInput(p1Color1.text) || !ValidateInput(p1Color2.text) || !ValidateInput(p1Color3.text) ||
            !ValidateInput(p2Card1.text) || !ValidateInput(p2Card2.text) || !ValidateInput(p2Card3.text) ||
            !ValidateInput(p2Color1.text) || !ValidateInput(p2Color2.text) || !ValidateInput(p2Color3.text))
        {
            resultText.text = "Invalid input. Please enter valid numbers.";
            return;
        }

        try
        {
            int[] p1Cards = new int[]
            {
                int.Parse(p1Card1.text),
                int.Parse(p1Card2.text),
                int.Parse(p1Card3.text)
            };
            int[] p1Colors = new int[]
            {
                int.Parse(p1Color1.text),
                int.Parse(p1Color2.text),
                int.Parse(p1Color3.text)
            };
            int[] p2Cards = new int[]
            {
                int.Parse(p2Card1.text),
                int.Parse(p2Card2.text),
                int.Parse(p2Card3.text)
            };
            int[] p2Colors = new int[]
            {
                int.Parse(p2Color1.text),
                int.Parse(p2Color2.text),
                int.Parse(p2Color3.text)
            };

            modelCommunicator.SendDataToModel(p1Cards, p1Colors, p2Cards, p2Colors, OnModelPrediction);
        }
        catch (System.Exception e)
        {
            resultText.text = $"Error parsing input: {e.Message}";
            Debug.LogError($"Error parsing input: {e.Message}");
        }
    }

    private bool ValidateInput(string input)
    {
        int number;
        return int.TryParse(input, out number);
    }

    void OnModelPrediction(int result)
    {
        switch (result)
        {
            case 0:
                resultText.text = "Player 1 wins!";
                break;
            case 1:
                resultText.text = "Player 2 wins!";
                break;
            case 2:
                resultText.text = "It's a draw!";
                break;
        }
    }
}
