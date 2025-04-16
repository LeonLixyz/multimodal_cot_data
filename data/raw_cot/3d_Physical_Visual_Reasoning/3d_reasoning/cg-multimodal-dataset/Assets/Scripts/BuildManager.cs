using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.Collections;
using UnityEngine;
using Random = System.Random;

public class BuildManager : MonoBehaviour
{
    private Dictionary<int, string> displayStageString = new Dictionary<int, string>()
    {
        //{ 0, "Welcome. \n Press [Space] to generate new scene." },
        { 1, "Generated Scene. \n Press[Space] to process." },
        { 2, "Eliminated. \n Press [Space] to regenerate scene." }
    };
    [Tooltip("0 - None, 1 - Generated Scene, 2- Processed Scene")]
    public int displayStage = 1;
    
    // SERIALIZED PRIVATE VARIABLES
    [Header("References")]
    [SerializeField] private GameObject entityPrefab;
    [SerializeField] private Transform planeTopLeft;
    [SerializeField] private Transform planeBotRight;
    [SerializeField] private TMP_Text screenText;
    
    
    // PRIVATE VARIABLES
    private EntityGroupProcessor entityGroup;

    private void Awake()
    {
        entityGroup = new EntityGroupProcessor();
    }

    private void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            displayStage++;
            if (displayStage > 2)
            {
                displayStage = 1;
            }

            if (displayStage == 1)
            {
                GenerateNewScene(UnityEngine.Random.Range(8,15));
            }
            else if (displayStage == 2)
            {
                ProcessScene();
            }
        }
    }

    void GenerateNewScene(int entityAmount)
    {
        entityGroup.Clear();

        foreach (Transform child in transform)
        {
            Destroy(child.gameObject);
        }

        Vector3 topLeft = planeTopLeft.position;
        Vector3 botRight = planeBotRight.position;

        List<Vector3> placedPositions = new List<Vector3>();
        int attempts = 0;
        int maxAttempts = entityAmount * 20; // 防止无限循环

        while (placedPositions.Count < entityAmount && attempts < maxAttempts)
        {
            attempts++;

            float randX = UnityEngine.Random.Range(topLeft.x, botRight.x);
            float randZ = UnityEngine.Random.Range(botRight.z, topLeft.z);
            Vector3 spawnPos = new Vector3(randX, topLeft.y, randZ);

            bool tooClose = false;
            foreach (var pos in placedPositions)
            {
                if (Vector3.Distance(spawnPos, pos) < 1.6f)
                {
                    tooClose = true;
                    break;
                }
            }

            if (tooClose) continue;

            GameObject entity = Instantiate(entityPrefab, spawnPos, Quaternion.identity, transform);
            Entity entityScript = entity.GetComponent<Entity>();
            entityScript.Initialize();
            entityGroup.AddEntity(entityScript);

            placedPositions.Add(spawnPos);
        }

        if (placedPositions.Count < entityAmount)
        {
            Debug.LogWarning($"Only placed {placedPositions.Count}/{entityAmount} entities due to space constraints.");
        }

        // 更新UI
        if (screenText != null && displayStageString.ContainsKey(displayStage))
        {
            screenText.text = displayStageString[displayStage];
        }
    }


    void ProcessScene()
    {
        string displayText = entityGroup.ProcessEntities();
        
        if (screenText != null )
        {
            screenText.text = displayText;
        }
    }
}
