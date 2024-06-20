using UnityEngine;
using UnityEngine.UI;
using TMPro;  

public class GameManager : MonoBehaviour
{
    public TMP_InputField p1Card1;
    public TMP_InputField p1Card2;
    public TMP_InputField p1Card3;
    public TMP_InputField p2Card1;
    public TMP_InputField p2Card2;
    public TMP_InputField p2Card3;
    public TextMeshProUGUI resultText;
    [SerializeField] private ModelCommunicator modelCommunicator;


    void Start()
    {
        modelCommunicator = GetComponent<ModelCommunicator>();
    }

    public void OnSubmit()
    {
        int[] p1Cards = new int[]
        {
            int.Parse(p1Card1.text),
            int.Parse(p1Card2.text),
            int.Parse(p1Card3.text)
        };
        int[] p2Cards = new int[]
        {
            int.Parse(p2Card1.text),
            int.Parse(p2Card2.text),
            int.Parse(p2Card3.text)
        };

        modelCommunicator.SendDataToModel(p1Cards, p2Cards, OnModelPrediction);
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
