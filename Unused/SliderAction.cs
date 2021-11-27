using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class SliderAction : MonoBehaviour
{
    [SerializeField] public Slider _slider;
    [SerializeField] public TextMeshProUGUI _sliderText;
    // Start is called before the first frame update
    void Start()
    {
        _slider.onValueChanged.AddListener((v) =>
        {
            _sliderText.text = v.ToString("0.00");

            GameObject terrain = GameObject.Find("TerrainLoader");
            MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
            meshGenerator.AdjustScale(v);
        });
    }

}
