using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ScaleTerrain : MonoBehaviour
{
    [SerializeField] public float scaleValue = 0;
    [SerializeField] public TextMeshProUGUI _valueText;
    [SerializeField] public TextMeshProUGUI _loadingText;

    // Start is called before the first frame update
    void Start()
    {
            
    }
    void Update()
    {

            if (Input.GetKey(KeyCode.M))
            {
            scaleValue++;
            _valueText.text = scaleValue.ToString();
            GameObject terrain = GameObject.Find("TerrainLoader");
            MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
            meshGenerator.AdjustScale(0.01f);
        }
        if (Input.GetKey(KeyCode.N))
        {
            scaleValue--;
            _valueText.text = scaleValue.ToString();
            GameObject terrain = GameObject.Find("TerrainLoader");
            MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
            meshGenerator.AdjustScale(-0.01f);
        }

    }

}

