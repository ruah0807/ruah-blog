import React, { useState } from 'react';
import Header from './components/Header';
import PostList from './pages/PostList';
import MDFile from './pages/MDFile';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';

const App: React.FC = () => {

  const [selectedFile, setSelectedFile]  = useState<string>('');
  const [showSidebar, setShowSidebar] = useState<boolean>(false);

  const handleSelectFile = (fileName: string) => {
    setSelectedFile(fileName);
    setShowSidebar(false);
    console.log(`Selected file: ${fileName}`);
  };

  const toggleSidebar = () => {
    setShowSidebar(!showSidebar);
  }

  return (
    // <Router basename={process.env.PUBLIC_URL}>  
    <Router>
      <Header toggleSidebar={toggleSidebar} />
      <div className={`container ${showSidebar ? 'show-sidebar' : ''}`}>
        <div className='content-wrapper'>
          <div className='post-list-container'>
            <PostList onSelect={handleSelectFile} />
          </div>
          <div className='post-container'>
            <Routes>
                <Route path="/" element={<div className='main-post'>롸그에 오신것을 환영합니다.</div>} />
                <Route path="/:subtitle" element={<MDFile fileName={selectedFile} />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;