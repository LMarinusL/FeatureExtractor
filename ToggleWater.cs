using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;



public class ToggleWater : MonoBehaviour
{
    public float height;
    [SerializeField] public TextMeshProUGUI _valueText;
    [SerializeField] public GameObject water;
    // Start is called before the first frame update
    void Start()
    {

    }
    void Update()
    {
        height = Mathf.Round(transform.position.y);
        _valueText.text = height.ToString();
        if (Input.GetKey(KeyCode.B))
        {
            transform.position += new Vector3(0, 1, 0); ;
        }
        if (Input.GetKey(KeyCode.V))
        {
            transform.position += new Vector3(0, -1, 0); ;
        }


    }

}
