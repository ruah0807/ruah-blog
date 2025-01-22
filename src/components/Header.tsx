import React from 'react';
import './Header.css';

interface HeaderProps {
  toggleSidebar: () => void;
}
const Header: React.FC<HeaderProps> = ({ toggleSidebar }) => {
  const handleTitleClick = () => {
    window.location.href = '/'; // Navigate to the main page
  };

  return (
    <header>
      <h1 className='header-title' onClick={handleTitleClick} style={{ cursor: 'pointer' }}>
        롸그
        </h1>
      <nav>
        <ul>
          <li>About Me</li>
        <li className='hamburger' onClick={toggleSidebar}>
          ☰
        </li>
        </ul>
      </nav>
    </header>
  );
};

export default Header; 