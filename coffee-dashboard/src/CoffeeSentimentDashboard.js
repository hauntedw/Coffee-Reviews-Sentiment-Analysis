import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import './styles.css'; // Import the CSS file

// Your JSON data
const jsonData = {
  "topDescriptors": [
    { "name": "sweet", "value": 172 },
    { "name": "chocolate", "value": 89 },
    { "name": "crisp", "value": 85 },
    { "name": "cocoa", "value": 79 },
    { "name": "smooth", "value": 75 },
    { "name": "rich", "value": 72 },
    { "name": "bright", "value": 71 },
    { "name": "floral", "value": 64 }
  ],
  "sentimentByOrigin": [
    {
      "name": "Sidamo growing region, Southern Ethiopia",
      "reviewCount": 15,
      "averageSentiment": 0.83,
      "belowMedian": 0.53,
      "aboveMedian": 0.47,
      "bottom25": 0.33,
      "midLower25": 0.2,
      "midUpper25": 0.2,
      "top25": 0.27
    },
    {
      "name": "Guji Zone, Oromia Region, southern Ethiopia",
      "reviewCount": 12,
      "averageSentiment": 0.87,
      "belowMedian": 0.42,
      "aboveMedian": 0.58,
      "bottom25": 0.42,
      "midLower25": 0.0,
      "midUpper25": 0.33,
      "top25": 0.25
    },
    {
      "name": "Ethiopia",
      "reviewCount": 10,
      "averageSentiment": 0.86,
      "belowMedian": 0.8,
      "aboveMedian": 0.2,
      "bottom25": 0.4,
      "midLower25": 0.4,
      "midUpper25": 0.2,
      "top25": 0.0
    },
    {
      "name": "Guji Zone, Oromia region, Southern Ethiopia",
      "reviewCount": 7,
      "averageSentiment": 0.93,
      "belowMedian": 0.29,
      "aboveMedian": 0.71,
      "bottom25": 0.14,
      "midLower25": 0.14,
      "midUpper25": 0.14,
      "top25": 0.57
    },
    {
      "name": "Tarrazu, Costa Rica",
      "reviewCount": 6,
      "averageSentiment": 0.93,
      "belowMedian": 0.0,
      "aboveMedian": 1.0,
      "bottom25": 0.0,
      "midLower25": 0.0,
      "midUpper25": 1.0,
      "top25": 0.0
    }
  ],
  "sentimentDistribution": [
    { "name": "Very Negative (-1.0 to -0.5)", "count": 1 },
    { "name": "Negative (-0.5 to 0)", "count": 3 },
    { "name": "Neutral (0 to 0.5)", "count": 5 },
    { "name": "Positive (0.5 to 0.75)", "count": 12 },
    { "name": "Very Positive (0.75 to 0.85)", "count": 25 },
    { "name": "Excellent (0.85 to 0.95)", "count": 93 },
    { "name": "Outstanding (0.95 to 1.0)", "count": 65 }
  ],
  "radarData": [
    { "descriptor": "sweet", "frequency": 172 },
    { "descriptor": "chocolate", "frequency": 89 },
    { "descriptor": "crisp", "frequency": 85 },
    { "descriptor": "cocoa", "frequency": 79 },
    { "descriptor": "smooth", "frequency": 75 },
    { "descriptor": "rich", "frequency": 72 },
    { "descriptor": "bright", "frequency": 71 },
    { "descriptor": "floral", "frequency": 64 }
  ]
};

// Transform the data to match the expected format
const transformedData = {
  // topDescriptors format matches, so we can use it directly
  topDescriptors: jsonData.topDescriptors,
  
  // Transform sentimentByOrigin to match the expected format with positive and negative values
  sentimentByOrigin: jsonData.sentimentByOrigin.map(item => ({
    // Use shorter display names rather than truncation
    name: item.name,
    fullName: item.name, // Keep full name for tooltips
    positive: item.averageSentiment,
    negative: 1 - item.averageSentiment, // Convert to negative sentiment
    reviewCount: item.reviewCount // Add review count for tooltip
})).sort((a, b) => b.positive - a.positive), // Sort by positive sentiment
  
  // sentimentDistribution format matches, so we can use it directly
  sentimentDistribution: jsonData.sentimentDistribution,
  
  // Transform radarData to match the expected format
  radarData: jsonData.radarData.map(item => ({
    descriptor: item.descriptor,
    frequency: item.frequency
  }))
};

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1'];

const CoffeeSentimentDashboard = () => {
  const [activeTab, setActiveTab] = useState('descriptors');
  
  // Generate insights based on the actual data
  const generateDescriptorInsights = () => {
    const topDescriptor = transformedData.topDescriptors[0];
    const totalMentions = transformedData.topDescriptors.reduce((sum, item) => sum + item.value, 0);
    const top3Total = transformedData.topDescriptors.slice(0, 3).reduce((sum, item) => sum + item.value, 0);
    const top3Percentage = Math.round((top3Total / totalMentions) * 100);
    
    return (
      <ul>
        <li>{topDescriptor.name} notes are the most commonly mentioned flavor descriptor in coffee reviews</li>
        <li>The second and third most common descriptors are {transformedData.topDescriptors[1].name} and {transformedData.topDescriptors[2].name}</li>
        <li>The top 3 flavors account for approximately {top3Percentage}% of all flavor mentions</li>
      </ul>
    );
  };
  
  const generateSentimentInsights = () => {
    const highestSentiment = [...transformedData.sentimentByOrigin].sort((a, b) => b.positive - a.positive)[0];
    const mostPositiveRange = [...transformedData.sentimentDistribution].sort((a, b) => b.count - a.count)[0];
    
    return (
      <ul>
        <li>{highestSentiment.name} coffees have the highest positive sentiment in reviews at {(highestSentiment.positive * 100).toFixed(1)}%</li>
        <li>Most coffee reviews ({mostPositiveRange.count} reviews) have a sentiment score in the {mostPositiveRange.name} range</li>
        <li>Reviews mentioning "{transformedData.topDescriptors[0].name}" and "{transformedData.topDescriptors[3].name}" tend to have higher sentiment scores</li>
        <li>Only {transformedData.sentimentDistribution[0].count + transformedData.sentimentDistribution[1].count} reviews fall below neutral sentiment, suggesting few negative reviews</li>
      </ul>
    );
  };
  
  return (
    <div className="dashboard">
      <header className="header">
        <h1>Coffee Review Sentiment Analysis</h1>
      </header>
      
      <nav className="nav">
        <ul>
          <li>
            <button 
              className={activeTab === 'descriptors' ? 'active' : ''}
              onClick={() => setActiveTab('descriptors')}
            >
              Flavor Descriptors
            </button>
          </li>
          <li>
            <button 
              className={activeTab === 'sentiment' ? 'active' : ''}
              onClick={() => setActiveTab('sentiment')}
            >
              Sentiment Analysis
            </button>
          </li>
          <li>
            <button 
              className={activeTab === 'radar' ? 'active' : ''}
              onClick={() => setActiveTab('radar')}
            >
              Flavor Profile
            </button>
          </li>
        </ul>
      </nav>
      
      <main className="main">
        {activeTab === 'descriptors' && (
          <div className="card">
            <h2>Top Coffee Flavor Descriptors</h2>
            <div style={{ height: '24rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={transformedData.topDescriptors}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 80, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" name="Frequency" fill="#8b4513" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="insights">
              <h3>Insights:</h3>
              {generateDescriptorInsights()}
            </div>
          </div>
        )}
        
        {activeTab === 'sentiment' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card">
              <h2>Sentiment by Origin</h2>
              <div style={{ height: '24rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={transformedData.sentimentByOrigin}
                    margin={{ top: 20, right: 30, left: 20, bottom: 90 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} interval={0} tick={{ fontSize: 10 }} /> {/* Angled labels */}
                    <YAxis />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="custom-tooltip" style={{ backgroundColor: 'white', padding: '10px', border: '1px solid #ccc' }}>
                              <p className="label">{payload[0].payload.fullName}</p>
                              <p style={{ color: '#82ca9d' }}>
                                Positive Sentiment: {(payload[0].value * 100).toFixed(1)}%
                              </p>
                              <p style={{ color: '#ff7675' }}>
                                Negative Sentiment: {(payload[1].value * 100).toFixed(1)}%
                              </p>
                              <p>Total Reviews: {payload[0].payload.reviewCount}</p>
                            </div>
                          );
                        }
                        return null;
                      }} 
                    />
                    <Legend verticalAlign='top' height={36} />
                    <Bar dataKey="positive" name="Positive Sentiment" stackId="a" fill="#82ca9d" />
                    <Bar dataKey="negative" name="Negative Sentiment" stackId="a" fill="#ff7675" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div className="card">
              <h2>Sentiment Distribution</h2>
              <div style={{ height: '24rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={transformedData.sentimentDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                      nameKey="name"
                      label={({name, percent}) => `${name.length > 10 ? name.substring(0, 10) + '...' : name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {transformedData.sentimentDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value, name, props) => [value, props.payload.name]} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div className="card md:col-span-2">
              <h3>Sentiment Analysis Insights:</h3>
              {generateSentimentInsights()}
            </div>
          </div>
        )}
        
        {activeTab === 'radar' && (
          <div className="card">
            <h2>Coffee Flavor Profile Radar</h2>
            <div style={{ height: '20rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={transformedData.radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="descriptor" />
                  <PolarRadiusAxis angle={30} domain={[0, 200]} />
                  <Radar name="Flavor Profile" dataKey="frequency" stroke="#8b4513" fill="#8b4513" fillOpacity={0.6} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="insights">
              <h3>Flavor Profile Insights:</h3>
              <ul>
                <li>The radar chart visualizes the distribution of flavor notes in the reviewed coffees</li>
                <li>There's a clear preference for {transformedData.radarData[0].descriptor}, {transformedData.radarData[1].descriptor}, and {transformedData.radarData[2].descriptor} flavor profiles</li>
                <li>The shape of the radar indicates a balanced distribution across major flavor categories</li>
                <li>{transformedData.radarData[0].descriptor} appears {Math.round(transformedData.radarData[0].frequency / transformedData.radarData[7].frequency)} times more frequently than {transformedData.radarData[7].descriptor}</li>
              </ul>
            </div>
          </div>
        )}
      </main>
      
      <footer className="footer">
        <p>Coffee Review Sentiment Analysis Dashboard</p>
      </footer>
    </div>
  );
};

export default CoffeeSentimentDashboard;